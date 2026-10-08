"""RGB order, fixed FP32 preprocessing, and whole-model graph capture."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.models.mdd_pose import (
    MDDPoseConfig,
    MDDPoseDetector,
    MDDQueryDetector,
)
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.training.coordinate_checkpoint import (
    validate_coordinate_checkpoint,
)


@pytest.mark.parametrize("a,b", [(.2, .15), (.27, -.05)])
def test_native_mdd_matches_old_numpy_bgr_and_stays_fp32_under_autocast(a: float, b: float) -> None:
    torch.set_num_threads(2)
    bgr = np.random.default_rng(12).integers(0, 256, (32, 16, 17, 3), dtype=np.uint8)
    rgb = torch.from_numpy(bgr[..., ::-1].transpose(0, 3, 1, 2).copy())[None]
    image = bgr.astype(np.float32) / 255
    gray = .114 * image[..., 0] + .587 * image[..., 1] + .299 * image[..., 2]
    transform = RGBToMDD(a, b)
    diff = np.diff(gray, axis=0)
    expected: NDArray[np.float32] = np.zeros((1, 2, 32, 16, 17), np.float32)
    for channel, delta in enumerate((diff, -diff)):
        logits = np.clip((np.maximum(delta, 0) - transform.offset) * transform.gain, -80, 80)
        expected[0, channel, 1:] = 1 / (1 + np.exp(-logits))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = transform(rgb)
    assert actual.dtype == torch.float32 and not actual[:, :, 0].any()
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=2e-6, atol=2e-7)
    assert not torch.allclose(actual, transform(rgb.flip(2)))
    assert list(transform.parameters()) == []
    assert RGBToMDD.from_contract(transform.input_contract()).input_contract() == transform.input_contract()


@pytest.mark.parametrize("violation", ["dtype", "layout", "legacy_mdd", "contract"])
def test_native_input_contract_rejects_ambiguous_inputs(violation: str) -> None:
    transform = RGBToMDD()
    rgb = torch.zeros(1, 32, 3, 16, 16, dtype=torch.uint8)
    with pytest.raises(ValueError):
        if violation == "contract":
            RGBToMDD.from_contract(dict(transform.input_contract(), color_order="BGR"))
        elif violation == "dtype":
            transform(rgb.float())
        elif violation == "layout":
            transform(rgb.transpose(1, 2))
        else:
            transform(torch.zeros(1, 2, 32, 16, 16))


@pytest.mark.parametrize("pose", [False, True])
def test_whole_native_forward_has_no_graph_break_and_preserves_state_names(pose: bool) -> None:
    torch.set_num_threads(2)
    cfg = MDDPoseConfig("conv2d", None, "query_only", 32, (4, 4, 8, 8), (8, 8), 16, 2, 1, 0., 10000.)
    model = MDDPoseDetector(replace(cfg, pose_pooling="attention", readout="query")) if pose else MDDQueryDetector(cfg)
    rgb = torch.randint(256, (1, 32, 3, 16, 16), dtype=torch.uint8)
    times = torch.arange(32)[None].float() / 30
    args = (rgb, torch.rand(1, 32, 2, 17, 2), torch.ones(1, 32, 2, 17, dtype=torch.bool), times) if pose else (rgb, times)
    expected = model(*args)
    names = set(model.state_dict())
    # CPU graph capture checks structure; GPU Inductor/autocast is measured separately.
    model.compile(backend="eager", fullgraph=True, dynamic=False)
    actual = model(*args)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert set(model.state_dict()) == names
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_old_mdd_checkpoint_is_not_silently_loaded_as_rgb() -> None:
    with pytest.raises(ValueError, match="legacy MDD-input v2"):
        validate_coordinate_checkpoint({"schema": "mdd_coordinates.v2"})
