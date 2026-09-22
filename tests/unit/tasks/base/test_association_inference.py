"""Confidence rejection and the raw observation -> tracked -> raw boundary."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.data.association_observations import (
    prepare_association_observations,
)
from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.inference.association import predict_association_observations
from src.tasks.base.model_io.association_contracts import (
    AssociationInferencePolicy,
    AssociationObservationRequest,
    IdentityReason,
)
from src.tasks.base.model_io.association_decoding import decode_reliable_identities


def tracking() -> ObservationTrackingConfig:
    return ObservationTrackingConfig(0.08, 8, 4, True, 4, "median", "error")


def request() -> AssociationObservationRequest:
    uv = torch.zeros(3, 3, 2, 17, 2)
    uv[:, :, 0] = 0.2
    uv[:, :, 1] = 0.8
    return AssociationObservationRequest(
        uv, torch.ones(uv.shape[:-1], dtype=torch.bool),
        torch.full((3, 3, 14, 2), 0.5), torch.ones(3, 3, 14, dtype=torch.bool),
        ("left", "right", "center"), "right", torch.tensor([0, 2, 4]),
    )


def test_uniform_logits_and_symmetric_collision_are_unknown() -> None:
    policy = AssociationInferencePolicy()
    for logits in (torch.zeros(1, 4, 11), torch.tensor([[[8.] + [-8.] * 10] * 4])):
        decoded = decode_reliable_identities(logits, torch.ones(1, 4, dtype=torch.bool), policy=policy)
        assert decoded["ids"].tolist() == [[-1] * 4]
        assert decoded["reasons"].eq(int(IdentityReason.AMBIGUOUS)).all()


def test_confident_ids_fp_and_missing_are_distinct() -> None:
    logits = torch.full((1, 4, 11), -8.)
    logits[0, 0, 0] = logits[0, 1, 3] = logits[0, 2, 10] = 8
    out = decode_reliable_identities(logits, torch.tensor([[True, True, True, False]]), policy=AssociationInferencePolicy())
    assert out["ids"].tolist() == [[0, 3, -1, -1]]
    assert out["reasons"].tolist() == [[0, 0, 2, 1]]


def test_padding_and_mapping_preserve_raw_camera_local_observations() -> None:
    req = request()
    policy = AssociationInferencePolicy()
    inputs, indices = prepare_association_observations(req, tracking=tracking(), policy=policy, joints=17)
    assert inputs["object_uv"].shape == (1, 5, 512, 4, 17, 2)
    assert inputs["padding_mask"][0, :3, :3].eq(False).all()
    assert inputs["padding_mask"][0, 3:].all()
    assert inputs["padding_mask"][0, :, 3:].all()
    assert inputs["reference_view_index"].tolist() == [1]
    assert torch.equal(inputs["court_kp"][0, :3, :3], req.court_kp)
    for v, t, slot in (indices >= 0).nonzero().tolist():
        assert torch.equal(inputs["object_uv"][0, v, t, slot], req.object_uv[v, t, indices[v, t, slot]])


def test_raw_ids_are_restored_through_detection_indices_once() -> None:
    calls = []
    def predict(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calls.append(inputs)
        shape = inputs["object_uv"].shape[:4]
        logits = torch.full((*shape, 11), -8.)
        for slot in range(4):
            logits[..., slot, slot] = 8
        return {"object_id_logits": logits, "side_logits": torch.zeros(1, 5), "view_half_turns": torch.zeros(1, 5, dtype=torch.bool)}
    out = predict_association_observations(predict, request(), tracking=tracking(), policy=AssociationInferencePolicy(), joints=17)
    assert len(calls) == 1
    assert out.raw_ids.shape == (3, 3, 2)
    assert out.raw_ids.ge(0).all()
    assert out.raw_reasons.eq(0).all()
    assert out.frame_indices.tolist() == [0, 2, 4]


def test_no_implicit_clip_truncation() -> None:
    with pytest.raises(ValueError, match="no implicit"):
        prepare_association_observations(request(), tracking=tracking(), policy=AssociationInferencePolicy(min_frames=1, max_frames=2), joints=17)


def test_empty_observations_never_load_an_association_checkpoint(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.components.view_association import (
        ViewAssociationModule,
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("No checkpoint load for zero observations")
    monkeypatch.setattr("src.tasks.plcs.inference.association_predictor.PLCSAssociationPredictor.load", forbidden)
    module = ViewAssociationModule(tmp_path / "absent.ckpt", task="plcs", device="cpu")
    req = request()
    req = replace(req, object_vis=torch.zeros_like(req.object_vis))
    assert module.process_observations(req, policy=AssociationInferencePolicy()) is None
