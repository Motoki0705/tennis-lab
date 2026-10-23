from dataclasses import replace

import pytest
import torch

from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
    transform_uv,
)
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.preprocessing.rgb_mdd import (
    RGBToRGBMDD,
    input_transform_from_contract,
)
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("step", [1, 2, 4])
def test_five_channels_use_sampled_rgb_and_exact_fp32_mdd(step: int) -> None:
    torch.set_num_threads(2)
    rgb = torch.randint(256, (2, 128, 3, 9, 13), dtype=torch.uint8)
    sampled = rgb[:, 1:1 + step * 32:step]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = RGBToRGBMDD()(sampled)
    assert actual.shape == (2, 5, 32, 9, 13) and actual.dtype == torch.float32
    torch.testing.assert_close(actual[:, :3], sampled.transpose(1, 2).float() / 255, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 3:], RGBToMDD()(sampled), rtol=0, atol=0)
    assert not actual[:, 3:, 0].any()
    assert actual[:, :3, 0].any() and 0 <= actual.min() <= actual.max() <= 1
    if step > 1:
        assert not torch.equal(actual[:, 3:, 1:], RGBToMDD()(rgb)[:, :, 1:1 + step * 32:step][:, :, 1:])


@pytest.mark.parametrize("profile", ["camera", "occlusion", "combined"])
def test_augmented_rgb_mdd_and_gt_share_the_sampled_window(profile: str) -> None:
    torch.set_num_threads(2)
    cfg = AugmentationConfig.load(PROJECT_ROOT / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    augmenter = VideoAugmenter(replace(cfg, targeted_probability=1.), 42)
    rgb = torch.randint(256, (1, 128, 3, 32, 64), dtype=torch.uint8)[:, ::4]
    batch = dict(uv=torch.full((1, 32, 2), .5), position_valid=torch.ones(1, 32, dtype=torch.bool),
                 timestamps=torch.arange(32)[None].float() * 4 / 30, start=[0], frame_step=[4], clip_id=["clip"])
    augmented, labels, audit = augmenter(rgb, batch, epoch=0, profile=profile)
    actual = RGBToRGBMDD()(augmented)
    torch.testing.assert_close(actual[:, :3], augmented.transpose(1, 2).float() / 255, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 3:], RGBToMDD()(augmented), rtol=0, atol=0)
    torch.testing.assert_close(labels["uv"], transform_uv(batch["uv"], audit["matrix"]))
    assert labels["timestamps"] is batch["timestamps"]
    assert labels["position_valid"][audit["artificially_occluded"]].all()
    assert not torch.equal(actual[:, 3:], RGBToMDD()(rgb))


@pytest.mark.parametrize("field,value", [
    ("frames", 16), ("rgb_normalization", "imagenet"),
    ("encoder_channels", ["B", "G", "R", "MDD_positive", "MDD_negative"]),
    ("temporal_difference", "native_frames"), ("first_mdd_frame", "repeat"),
])
def test_contract_rejects_changed_semantics(field: str, value: object) -> None:
    contract = RGBToRGBMDD().input_contract()
    assert input_transform_from_contract(contract, input_channels=5).input_contract() == contract
    with pytest.raises(ValueError, match="input contract"):
        input_transform_from_contract(dict(contract, **{field: value}), input_channels=5)
    with pytest.raises(ValueError, match="input contract"):
        input_transform_from_contract(contract, input_channels=2)


@pytest.mark.parametrize("frames", [1, 7, 32, 64, 128, 256])
def test_explicit_temporal_contract_and_mdd_do_not_reset_inside_long_clip(frames: int) -> None:
    torch.set_num_threads(1)
    transform = RGBToRGBMDD(frames=frames)
    rgb = torch.randint(256, (1, frames, 3, 9, 13), dtype=torch.uint8)
    actual = transform(rgb)
    torch.testing.assert_close(actual[:, 3:], RGBToMDD()(rgb), rtol=0, atol=0)
    assert actual.shape == (1, 5, frames, 9, 13)
    contract = transform.input_contract()
    assert contract["frames"] == frames
    restored = input_transform_from_contract(contract, input_channels=5, frames=frames)
    torch.testing.assert_close(restored(rgb), actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="input contract"):
        input_transform_from_contract(contract, input_channels=5, frames=frames + 1)
