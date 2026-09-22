"""Padding-contract tests for the SLCS dataset collator."""

from __future__ import annotations

import pytest
import torch

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dataset import SLCSDataConfig, collate_slcs
from src.tasks.slcs.data.types import SLCSSample
from src.tennis_scene.generate_dataset.manifest import ClipManifest


def _sample(*, dino_samples: int, padding_mask: torch.Tensor) -> SLCSSample:
    players, frames, joints, court_kp = 1, 3, 2, 2
    real_frames = ~padding_mask
    return SLCSSample(
        player_kp=torch.zeros(players, frames, joints, 2),
        player_kp_vis=real_frames.view(1, frames, 1).expand(players, frames, joints),
        player_valid=real_frames.view(1, frames).expand(players, frames),
        ball_uv=torch.zeros(frames, 2),
        ball_vis=real_frames,
        court_kp=torch.zeros(frames, court_kp, 2),
        court_vis=real_frames.view(frames, 1).expand(frames, court_kp),
        dino_tokens=torch.ones(dino_samples, 2, 3),
        dino_frame_idx=torch.arange(dino_samples, dtype=torch.int64),
        dino_padding_mask=torch.zeros(dino_samples, dtype=torch.bool),
        frame_idx=torch.arange(frames, dtype=torch.int64),
        timestamp=torch.arange(frames, dtype=torch.float32),
        padding_mask=padding_mask,
        target_player_position=torch.zeros(players, frames, 3),
        target_player_rotation=torch.ones(players, frames, 2),
        target_player_valid=real_frames.view(1, frames).expand(players, frames),
        target_player_weight=real_frames.view(1, frames)
        .expand(players, frames)
        .float(),
        target_ball_position=torch.zeros(frames, 3),
        target_ball_valid=real_frames,
        target_ball_weight=real_frames.float(),
    )


def test_collate_preserves_frame_padding_polarity_and_pads_sparse_dino() -> None:
    first_padding = torch.tensor([False, False, True])
    second_padding = torch.zeros(3, dtype=torch.bool)

    batch = collate_slcs(
        [
            _sample(dino_samples=2, padding_mask=first_padding),
            _sample(dino_samples=0, padding_mask=second_padding),
        ]
    )

    torch.testing.assert_close(
        batch["padding_mask"], torch.stack([first_padding, second_padding])
    )
    assert batch["padding_mask"].dtype == torch.bool
    assert batch["dino_tokens"].shape == (2, 2, 2, 3)
    assert torch.equal(
        batch["dino_padding_mask"],
        torch.tensor([[False, False], [True, True]]),
    )
    assert not batch["dino_tokens"][1].any()


def test_collate_uses_one_explicit_padding_slot_when_every_dino_axis_is_empty() -> None:
    batch = collate_slcs(
        [
            _sample(
                dino_samples=0,
                padding_mask=torch.zeros(3, dtype=torch.bool),
            )
        ]
    )

    assert batch["dino_tokens"].shape == (1, 1, 2, 3)
    assert batch["dino_padding_mask"].tolist() == [[True]]


@pytest.mark.parametrize("name", ["human_kp_vis", "court_vis"])
@pytest.mark.parametrize("value", [1.03125, -0.01, float("nan"), float("inf")])
def test_load_rejects_invalid_visibility_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
    data_config: SLCSDataConfig,
    synthetic_dataset: SLCSDataIndex,
    name: str,
    value: float,
) -> None:
    import numpy as np

    from src.tasks.slcs.data import dataset
    from src.tennis_scene.generate_dataset.manifest import DatasetManifestError
    from tests.support.tasks.slcs.dataset import (
        SLCSFixtureDatasetConfig,
        make_fixture_scene,
    )

    scene = make_fixture_scene(SLCSFixtureDatasetConfig(), np.random.default_rng(0))
    getattr(scene, name).flat[0] = value
    monkeypatch.setattr(dataset, "load_slcs_annotation", lambda *a, **kw: scene)
    with pytest.raises(
        DatasetManifestError, match=rf"{name} must contain finite values in \[0, 1\]"
    ):
        manifest = ClipManifest.load(
            synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
        )
        dataset.load_clip_arrays(manifest, config=data_config)
    actual = getattr(scene, name).flat[0]
    assert actual == value or (np.isnan(actual) and np.isnan(value))


def test_camera_local_targets_rotate_and_keep_raw_court(data_config, synthetic_dataset):
    from dataclasses import replace

    from src.tasks.slcs.data.dataset import build_window_sample, load_clip_arrays
    from src.tasks.slcs.data.windows import plan_windows

    manifest = ClipManifest.load(synthetic_dataset.clip_dir(synthetic_dataset.clips[0]))
    clip = load_clip_arrays(manifest, config=data_config)
    plan = plan_windows(
        clip.num_frames,
        window_size=data_config.window_size,
        stride=data_config.window_size,
    )[0]
    kwargs = dict(camera_index=0, plan=plan, dino_arrays=None, empty_dino_shape=(1, 1))
    physical = build_window_sample(clip, **kwargs)
    local = build_window_sample(
        replace(clip, camera_half_turns=(True,) * clip.court_kp.shape[0]), **kwargs
    )
    torch.testing.assert_close(local["court_kp"], physical["court_kp"])
    torch.testing.assert_close(local["court_vis"], physical["court_vis"])
    torch.testing.assert_close(
        local["target_player_position"],
        physical["target_player_position"].flip(0) * torch.tensor([-1.0, -1.0, 1.0]),
    )
    torch.testing.assert_close(
        local["target_ball_position"],
        physical["target_ball_position"] * torch.tensor([-1.0, -1.0, 1.0]),
    )
    torch.testing.assert_close(
        local["target_player_rotation"], -physical["target_player_rotation"].flip(0)
    )
    torch.testing.assert_close(local["player_kp"], physical["player_kp"].flip(0))
