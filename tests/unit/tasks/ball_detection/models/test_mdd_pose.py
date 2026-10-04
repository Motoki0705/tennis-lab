from itertools import product

import pytest
import torch

from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDPoseDetector
from src.tasks.ball_detection.models.mdd_pose.encoder import (
    MDDTokenEncoder,
    SpatialReduction,
)
from src.tasks.ball_detection.models.mdd_pose.pooling import PoseTokenizer


@pytest.mark.parametrize("method,pool,readout", list(product(
    ("conv3d", "average", "unshuffle", "haar"),
    ("deepsets", "attention", "hierarchical", "gnn"), ("query", "pose"),
)))
def test_every_ablation_has_finite_masked_gradients(method: str, pool: str, readout: str) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(12)
    config = MDDPoseConfig(method, pool, readout, 32, (4, 8, 8), 16, 2, 1, 0., 10000.)
    model = MDDPoseDetector(config)
    mdd = torch.rand(1, 2, 32, 16, 16)
    pose = torch.rand(1, 32, 2, 17, 2)
    valid = torch.ones(1, 32, 2, 17, dtype=torch.bool)
    valid[:, 8:16] = False
    times = torch.arange(32)[None].float() / 30
    prediction = model(mdd, pose, valid, times)
    target = torch.rand_like(prediction)
    supervision = torch.ones(1, 32, dtype=torch.bool)
    supervision[:, 8:16] = False
    target[:, 8:16] = float("nan")
    loss = coordinate_loss(prediction, target, supervision)
    loss.backward()
    assert prediction.shape == (1, 32, 2)
    assert torch.isfinite(loss)
    assert model.encoder.stages[0][1 if method != "conv3d" else 0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("method", ["conv3d", "average", "unshuffle", "haar"])
def test_encoder_is_spatial_eighth_and_temporally_local(method: str) -> None:
    torch.set_num_threads(2)
    model = MDDTokenEncoder(method, (4, 8, 8), 16).eval()
    x = torch.rand(1, 2, 32, 16, 24)
    changed = x.clone()
    changed[:, :, 0] += 4
    with torch.no_grad():
        a, b = model(x), model(changed)
    assert a.shape == (1, 32, 6, 16)
    torch.testing.assert_close(a[:, 4:], b[:, 4:], rtol=0, atol=0)
    assert not torch.equal(a[:, :4], b[:, :4])


def test_haar_retains_all_subbands_before_projection() -> None:
    x = torch.arange(16).float().reshape(1, 1, 1, 4, 4)
    y = SpatialReduction("haar")(x)
    ll, lh, hl, hh = y.unbind(1)
    phases = torch.stack((ll + lh + hl + hh, ll - lh + hl - hh,
                          ll + lh - hl - hh, ll - lh - hl + hh), 1) / 2
    restored = torch.nn.functional.pixel_shuffle(phases[:, :, 0], 2)[:, :, None]
    torch.testing.assert_close(x, restored)


@pytest.mark.parametrize("method", ["deepsets", "attention", "hierarchical", "gnn"])
def test_pose_pooling_is_person_order_invariant_and_ignores_missing(method: str) -> None:
    model = PoseTokenizer(method, 16, 2).eval()
    coords = torch.rand(1, 3, 3, 17, 2)
    valid = torch.ones(1, 3, 3, 17, dtype=torch.bool)
    valid[:, :, 2] = False
    masked = coords.clone()
    masked[:, :, 2] = 999
    with torch.no_grad():
        a = model(coords, valid)
        b = model(masked.flip(2), valid.flip(2))
        empty = model(coords[:, :, :0], valid[:, :, :0])
    torch.testing.assert_close(a, b)
    assert torch.isfinite(empty).all()


@pytest.mark.parametrize("violation", ["rgb", "frames", "time", "mask", "spatial"])
def test_typed_boundary_rejects_bad_inputs_before_model(violation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.ball_detection.model_io.mdd_pose import (
        MDDPoseInput,
        build_mdd_pose_detector,
    )

    config = MDDPoseConfig("conv3d", "attention", "query", 32, (4, 8, 8), 16, 2, 1, 0., 10000.)
    pair = build_mdd_pose_detector(config)
    def forbidden(*args: object) -> torch.Tensor:
        raise AssertionError("Invalid input entered the model")
    monkeypatch.setattr(pair.model, "forward", forbidden)
    mdd, pose = torch.zeros(1, 2, 32, 16, 16), torch.zeros(1, 32, 1, 17, 2)
    valid, times = torch.ones(1, 32, 1, 17, dtype=torch.bool), torch.arange(32)[None].float()
    if violation == "rgb":
        mdd = torch.zeros(1, 3, 32, 16, 16)
    elif violation == "frames":
        mdd = mdd[:, :, :31]
    elif violation == "time":
        times.zero_()
    elif violation == "mask":
        valid = valid.float()
    else:
        mdd = mdd[..., :15]
    with pytest.raises(ValueError):
        pair.run(MDDPoseInput(mdd, pose, valid, times))
