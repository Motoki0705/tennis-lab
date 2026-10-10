from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.model_io.mdd_query import (
    MDDQueryInput,
    build_mdd_query_detector,
)
from src.tasks.ball_detection.models.mdd_pose import (
    MDDPoseConfig,
    MDDPoseDetector,
    MDDQueryDetector,
)
from src.tasks.ball_detection.models.mdd_pose.query import QueryFusionBlock


def config(method: str = "conv2d") -> MDDPoseConfig:
    return MDDPoseConfig(method, None, "query_only", 32, (4, 4, 8, 8), (8, 8), 16, 2, 2, 0., 10000.)


@pytest.mark.parametrize("method", ["conv2d", "average", "unshuffle", "haar"])
def test_query_only_trains_without_pose_or_patch_update_parameters(method: str) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(26)
    pair = build_mdd_query_detector(config(method))
    times = torch.arange(32)[None].float() / 15
    prediction = pair.run(MDDQueryInput(torch.randint(256, (1, 32, 3, 32, 32), dtype=torch.uint8), times))
    target = torch.rand_like(prediction)
    valid = torch.ones(1, 32, dtype=torch.bool)
    valid[:, 5:10] = False
    target[~valid] = float("nan")
    loss = coordinate_loss(prediction, target, valid)
    loss.backward()
    assert prediction.shape == (1, 32, 2)
    assert torch.isfinite(prediction).all() and torch.isfinite(loss)
    assert not any("pose" in key or "to_image" in key for key in pair.model.state_dict())
    assert next(pair.model.encoder.stem.parameters()).grad is not None
    assert pair.model.ball_query.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in pair.model.parameters() if p.grad is not None)


def test_queries_read_same_time_images_then_mix_over_real_timestamps() -> None:
    torch.manual_seed(41)
    block = QueryFusionBlock(config()).eval()
    queries = torch.randn(1, 32, 16)
    patches = torch.randn(1, 32, 5, 16)
    original = patches.clone()
    changed = patches.clone()
    changed[:, 0, :, 0] += 4
    times = torch.arange(32)[None].float() / 30
    captures: list[torch.Tensor] = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captures.append(args[0].detach().clone())

    hook = block.temporal.register_forward_pre_hook(capture)
    with torch.no_grad():
        first = block(queries, patches, times)
        second = block(queries, changed, times)
        slower = block(queries, patches, times * 4)
    hook.remove()
    torch.testing.assert_close(patches, original, rtol=0, atol=0)
    assert not torch.allclose(captures[0][:, 0], captures[1][:, 0])
    torch.testing.assert_close(captures[0][:, 1:], captures[1][:, 1:], rtol=0, atol=0)
    assert not torch.allclose(first[:, 1:], second[:, 1:])
    assert not torch.allclose(first, slower)


@pytest.mark.parametrize("violation", ["mdd", "frames", "time", "dtype", "spatial"])
def test_query_boundary_validates_before_forward(violation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = build_mdd_query_detector(config())

    def forbidden(*args: object) -> torch.Tensor:
        raise AssertionError("Invalid data entered the model")

    monkeypatch.setattr(pair.model, "forward", forbidden)
    rgb = torch.zeros(1, 32, 3, 16, 16, dtype=torch.uint8)
    times = torch.arange(32)[None].float()
    if violation == "mdd":
        rgb = torch.zeros(1, 2, 32, 16, 16)
    elif violation == "frames":
        rgb = rgb[:, :31]
    elif violation == "time":
        times.zero_()
    elif violation == "dtype":
        rgb = rgb.float()
    else:
        rgb = rgb[..., :7]
    with pytest.raises(ValueError):
        pair.run(MDDQueryInput(rgb, times))


def test_pose_presence_is_explicit_in_config_and_model_selection() -> None:
    with pytest.raises(ValueError, match="pose_pooling: null"):
        replace(config(), pose_pooling="attention")
    with pytest.raises(ValueError, match="pose pooling"):
        replace(config(), readout="pose")
    with pytest.raises(ValueError, match="requires pose"):
        MDDPoseDetector(config())
    with pytest.raises(ValueError, match="query_only"):
        MDDQueryDetector(replace(config(), readout="query", pose_pooling="attention"))
    path = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs/model/mdd_query.yaml"
    shipped = MDDPoseConfig.load(path)
    assert shipped.readout == "query_only" and shipped.pose_pooling is None
