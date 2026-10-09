from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.training.posttraining.augmentation import (
    AugmentationConfig,
    VideoAugmenter,
    transform_uv,
    warp_rgb,
)


def batch() -> dict:
    return dict(uv=torch.full((1, 32, 2), .5), position_valid=torch.ones(1, 32, dtype=torch.bool),
                timestamps=torch.arange(32)[None].float() / 30, start=[0], frame_step=[1], clip_id=["clip"])


def test_camera_warp_moves_rgb_and_gt_in_the_same_direction() -> None:
    image = torch.zeros(1, 32, 3, 17, 33, dtype=torch.uint8)
    image[:, :, :, 8, 8] = 255
    matrix = torch.eye(3).expand(1, 32, 3, 3).clone()
    matrix[..., 0, 2] = .25
    out = warp_rgb(image, matrix)
    assert torch.equal(out[:, :, :, 8, 16], torch.full((1, 32, 3), 255, dtype=torch.uint8))
    uv = torch.tensor([.25, .5]).expand(1, 32, 2)
    torch.testing.assert_close(transform_uv(uv, matrix), torch.full((1, 32, 2), .5))
    torch.testing.assert_close(warp_rgb(image, torch.eye(3).expand(1, 32, 3, 3)), image)


def test_synthetic_occlusion_retains_teachers_is_temporally_bounded_and_reproducible() -> None:
    cfg = replace(AugmentationConfig.load(Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"), camera_probability=0., occlusion_probability=1., targeted_probability=1.,
                  occlusion_min_size=.2, occlusion_max_size=.2)
    augmenter = VideoAugmenter(cfg, 42)
    original = batch()
    original["position_valid"][0, 10] = False
    rgb = torch.full((1, 32, 3, 32, 64), 255, dtype=torch.uint8)
    out, target, audit = augmenter(rgb, original, epoch=2)
    again, _, second = augmenter(rgb, original, epoch=2)
    assert torch.equal(out, again) and torch.equal(audit["matrix"], second["matrix"])
    assert torch.equal(target["position_valid"], original["position_valid"])
    torch.testing.assert_close(target["uv"], original["uv"])
    assert not audit["artificially_occluded"][0, 10]
    assert 1 <= int(audit["artificially_occluded"].sum()) <= 6
    altered = (out != rgb).flatten(2).any(2)[0].nonzero().flatten()
    assert 2 <= len(altered) <= 6
    assert torch.equal(altered, torch.arange(int(altered[0]), int(altered[-1]) + 1))
    assert torch.equal(rgb, torch.full_like(rgb, 255))


def test_camera_transform_uses_real_timestamps_and_mdd_is_recomputed_after_rgb() -> None:
    torch.set_num_threads(2)
    augmenter = VideoAugmenter(replace(AugmentationConfig.load(Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"), camera_probability=1., occlusion_probability=0.), 52)
    rgb = torch.randint(256, (1, 32, 3, 32, 64), dtype=torch.uint8)
    original = batch()
    out, target, audit = augmenter(rgb, original, epoch=0)
    slow = dict(original, timestamps=original["timestamps"] * 4)
    _, _, slower = augmenter(rgb, slow, epoch=0)
    assert not torch.equal(audit["matrix"], slower["matrix"])
    torch.testing.assert_close(target["uv"], transform_uv(original["uv"], audit["matrix"]))
    assert not torch.equal(RGBToMDD()(out), RGBToMDD()(rgb))
    clean, labels, clean_audit = augmenter(rgb, original, epoch=20, profile="clean")
    assert clean is rgb and labels is original and clean_audit == {}


def test_excessive_camera_crop_is_explicitly_rejected_not_made_a_negative() -> None:
    cfg = replace(AugmentationConfig.load(Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"), camera_probability=1., occlusion_probability=0., pan_speed=100.,
                  max_translation=3., max_removed_fraction=0., roll_speed_degrees=0., log_zoom_speed=0., jitter_amplitude=0.)
    original = batch()
    image = torch.zeros(1, 32, 3, 16, 32, dtype=torch.uint8)
    _, target, audit = VideoAugmenter(cfg, 4)(image, original, epoch=0)
    assert audit["camera_rejected"].all()
    torch.testing.assert_close(audit["matrix"], torch.eye(3).expand(1, 32, 3, 3))
    assert target["position_valid"].all()
