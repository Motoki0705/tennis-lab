"""CPU regression against numeric fixtures produced before the replacement."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.vendor.gvhmr import vitpose
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import TopdownHeatmapSimpleHead
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps

REFERENCE_PATH = Path(__file__).with_name("fixtures") / "heatmap_reference.npz"
CASES = (
    "released_layout", "alternate_kernels", "extra_convs", "direct_projection",
    "identity_projection", "identity_head",
)


def _reference(case: str):
    with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
        prefix = case + "/"
        arrays = {key.removeprefix(prefix): archive[key] for key in archive.files if key.startswith(prefix)}
    config_values = json.loads(str(arrays.pop("config")))
    for name in ("num_deconv_filters", "num_deconv_kernels", "num_conv_kernels"):
        config_values[name] = tuple(config_values[name])
    config = ViTPoseHeadConfig(**config_values)
    state_dict = {
        key.removeprefix("state/"): torch.from_numpy(value)
        for key, value in arrays.items() if key.startswith("state/")
    }
    return config, state_dict, arrays


@pytest.mark.parametrize("case", CASES)
def test_legacy_checkpoint_heatmaps_and_udp_keypoints(case: str) -> None:
    config, state_dict, reference = _reference(case)
    model = vitpose.VitPoseModel(
        nn.Conv2d(3, config.in_channels, 1), TopdownHeatmapSimpleHead(config)
    ).eval()
    # The saved keys/shapes are from the old implementation, not a round trip.
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in state_dict.items()
    }
    loaded = model.load_state_dict(state_dict, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    with torch.inference_mode():
        heatmaps = model(torch.from_numpy(reference["images"]))
        averaged = (heatmaps[:1] + flip_heatmap_coco17(heatmaps[1:])) * 0.5
    np.testing.assert_allclose(heatmaps.numpy(), reference["heatmaps"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(averaged.numpy(), reference["flip_average"], rtol=1e-5, atol=1e-7)
    for name, value in (("heatmaps", heatmaps), ("flip_average", averaged)):
        count = value.shape[0]
        coords, scores = keypoints_from_heatmaps(
            value.numpy(), center=np.tile([[100., 200.]], (count, 1)),
            scale=np.tile([[0.9, 1.2]], (count, 1)), use_udp=True, kernel=3,
        )
        np.testing.assert_allclose(coords, reference[name + "_coordinates"], rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(scores, reference[name + "_confidence"], rtol=1e-5, atol=1e-7)


def test_builder_uses_official_huge_settings_and_loads_on_cpu(tmp_path, monkeypatch) -> None:
    config, state_dict, reference = _reference("released_layout")
    checkpoint = tmp_path / "pose.pth"
    torch.save({"state_dict": state_dict}, checkpoint)
    settings = {}

    def small_backbone(**kwargs):
        settings.update(kwargs)
        return nn.Conv2d(3, config.in_channels, 1)

    monkeypatch.setattr(vitpose, "ViT", small_backbone)
    model = vitpose.build_vitpose_huge(checkpoint, head_config=config).eval()
    # From ViTPose's Apache-2.0 ViTPose_huge_coco_256x192.py, pinned in the header.
    assert settings == {
        "img_size": (256, 192), "patch_size": 16, "embed_dim": 1280,
        "depth": 32, "num_heads": 16, "ratio": 1, "use_checkpoint": False,
        "mlp_ratio": 4, "qkv_bias": True, "drop_path_rate": 0.55,
    }
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    with torch.inference_mode():
        heatmaps = model(torch.from_numpy(reference["images"]))
    np.testing.assert_allclose(heatmaps.numpy(), reference["heatmaps"], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("defect", ("missing", "unexpected", "shape"))
def test_builder_rejects_incompatible_checkpoint(tmp_path, monkeypatch, defect: str) -> None:
    config, state_dict, _ = _reference("released_layout")
    if defect == "missing":
        del state_dict["keypoint_head.final_layer.bias"]
    elif defect == "unexpected":
        state_dict["unknown.weight"] = torch.zeros(1)
    else:
        state_dict["keypoint_head.final_layer.bias"] = torch.zeros(1)
    checkpoint = tmp_path / "incompatible.pth"
    torch.save({"state_dict": state_dict}, checkpoint)
    monkeypatch.setattr(vitpose, "ViT", lambda **kwargs: nn.Conv2d(3, config.in_channels, 1))
    with pytest.raises(RuntimeError, match={
        "missing": "Missing key", "unexpected": "Unexpected key", "shape": "size mismatch",
    }[defect]):
        vitpose.build_vitpose_huge(checkpoint, head_config=config)
