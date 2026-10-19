"""Stored-resolution, grid-coordinate and visibility contracts for overlay GIFs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.visualization.io.clip import (
    ClipSequence,
    load_clip_sequence,
)
from src.tasks.ball_detection.visualization.rendering.clip_renderer import (
    DrawStyle,
    render_overlay_frames,
)
from src.tasks.ball_detection.visualization.rendering.markers import (
    RingStyle,
    TrailStyle,
)
from src.utils.geometry.keypoints import denormalize_grid_keypoints
from tests.support.tasks.ball_detection.store import frame, write_store_clip

DRAW = DrawStyle(
    gt_radius=6, pred_radius=6, thickness=2,
    gt_color_rgb=(255, 96, 96), pred_color_rgb=(96, 255, 128),
    text_color_rgb=(245, 245, 245), muted_text_color_rgb=(168, 168, 168),
)
RING = RingStyle(radius=9, thickness=2, center_radius=2, glow_radius=3, glow_opacity=0.12)
TRAIL = TrailStyle(radius=4, length=8, thickness=2, opacity=0.55)


@pytest.mark.parametrize("width,height", [(512, 288), (960, 540), (256, 144)])
@pytest.mark.parametrize("coord", [(0.213, 0.734), (0.0, 0.0), (1.0, 1.0)])
def test_overlay_is_one_textless_panel_at_grid_coordinate(
    width: int, height: int, coord: tuple[float, float],
) -> None:
    original: np.ndarray = np.full((height, width, 3), 18, dtype=np.uint8)
    (actual,) = render_overlay_frames(
        frames_rgb=[original], pred_coords_normalized=[coord],
        pred_visibility=[True], draw=DRAW,
    )
    expected = original.copy()
    x, y = denormalize_grid_keypoints(np.asarray(coord), width, height)
    cv2.circle(
        expected, (round(float(x)), round(float(y))),
        max(1, round(6 * width / 512)), DRAW.pred_color_rgb,
        thickness=max(1, round(2 * width / 512)), lineType=cv2.LINE_AA,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original, np.full_like(original, 18))


def test_invisible_frame_is_unchanged_without_interpolation() -> None:
    original: np.ndarray = np.full((270, 480, 3), 42, dtype=np.uint8)
    frames = render_overlay_frames(
        frames_rgb=[original] * 3,
        pred_coords_normalized=[(0.2, 0.3), (float("nan"), float("nan")), (0.8, 0.7)],
        pred_visibility=[True, False, True], draw=DRAW,
    )
    assert np.any(frames[0] != original)
    np.testing.assert_array_equal(frames[1], original)
    assert np.any(frames[2] != original)


def test_mismatched_prediction_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="shorter"):
        render_overlay_frames(
            frames_rgb=[np.zeros((20, 30, 3), dtype=np.uint8)],
            pred_coords_normalized=[], pred_visibility=[False], draw=DRAW,
        )


def test_ring_has_small_center_thin_outline_and_faint_glow() -> None:
    original: np.ndarray = np.zeros((160, 960, 3), dtype=np.uint8)
    coord = (480 / 959, 80 / 159)
    (actual,) = render_overlay_frames(
        frames_rgb=[original], pred_coords_normalized=[coord], pred_visibility=[True],
        draw=DRAW, style="ring", ring=RING,
    )
    np.testing.assert_array_equal(actual[80, 480], DRAW.pred_color_rgb)
    np.testing.assert_array_equal(actual[80, 489], DRAW.pred_color_rgb)
    assert actual[80, 485].max() < 20  # Clear space between dot and ring.
    assert 0 < actual[80, 493].max() < 30  # Faint glow outside the outline.
    np.testing.assert_array_equal(actual[:50], original[:50])


def test_trail_fades_and_has_bounded_source_frame_history() -> None:
    original: np.ndarray = np.zeros((160, 960, 3), dtype=np.uint8)
    frames = render_overlay_frames(
        frames_rgb=[original] * 5,
        pred_coords_normalized=[(x / 959, 80 / 159) for x in (100, 200, 300, 400, 500)],
        pred_visibility=[True] * 5, draw=DRAW, style="trail", trail=TRAIL,
    )
    assert 0 < frames[-1][80, 150, 1] < frames[-1][80, 450, 1] < DRAW.pred_color_rgb[1]
    np.testing.assert_array_equal(frames[-1][80, 500], DRAW.pred_color_rgb)
    short = TrailStyle(radius=4, length=3, thickness=2, opacity=0.55)
    final = render_overlay_frames(
        frames_rgb=[original] * 5,
        pred_coords_normalized=[(x / 959, 80 / 159) for x in (100, 200, 300, 400, 500)],
        pred_visibility=[True] * 5, draw=DRAW, style="trail", trail=short,
    )[-1]
    assert final[80, 150].max() == 0
    assert final[80, 350].max() > 0


@pytest.mark.parametrize("stride", [1, 2])
def test_trail_breaks_at_missing_source_frame_even_when_skipped(stride: int) -> None:
    original: np.ndarray = np.zeros((160, 960, 3), dtype=np.uint8)
    coords = [(x / 959, 80 / 159) for x in (100, 200, 300, 400, 500)]
    coords[3] = (float("nan"), float("nan"))
    frames = render_overlay_frames(
        frames_rgb=[original] * 5, pred_coords_normalized=coords,
        pred_visibility=[True, True, True, False, True], draw=DRAW,
        style="trail", trail=TRAIL, frame_stride=stride,
    )
    assert frames[-1][80, 400].max() == 0  # No bridge across missing frame 3.
    np.testing.assert_array_equal(frames[-1][80, 500], DRAW.pred_color_rgb)
    if stride == 1:
        np.testing.assert_array_equal(frames[3], original)


@pytest.mark.parametrize("stride", [0, -1])
def test_render_stride_must_be_positive(stride: int) -> None:
    with pytest.raises(ValueError, match="frame_stride"):
        render_overlay_frames(
            frames_rgb=[], pred_coords_normalized=[], pred_visibility=[], draw=DRAW,
            frame_stride=stride,
        )


def test_render_resize_uses_stored_image_and_keeps_model_input_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_id = "tracknet/game1/Clip1"
    store_dir = write_store_clip(
        tmp_path / "store", clip_id, [frame(0), frame(1)], size=(640, 360),
    )
    # A sharp edge is lost in the 128px model input. Rendering must read the
    # stored image rather than enlarge the model input back to 320px.
    source: np.ndarray = np.zeros((360, 640, 3), dtype=np.uint8)
    source[:, 301:304] = 255
    monkeypatch.setattr(BallFrameStore, "read_bgr", lambda _store, _row: source.copy())

    def load(width: int | None) -> ClipSequence:
        return load_clip_sequence(
            store_dir=store_dir, clip_id=clip_id, sequence_length=2,
            image_size_hw=(64, 128), max_frames=2, normalize_imagenet=False,
            imagenet_mean=(0.485, 0.456, 0.406), imagenet_std=(0.229, 0.224, 0.225),
            render_width=width,
        )

    legacy = load(None)
    overlay = load(320)
    torch.testing.assert_close(legacy.model_images, overlay.model_images, rtol=0, atol=0)
    assert legacy.render_frames_rgb[0].shape == (64, 128, 3)
    assert overlay.render_frames_rgb[0].shape == (180, 320, 3)
    decoded = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    expected = cv2.resize(decoded, (320, 180), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(overlay.render_frames_rgb[0].numpy(), expected)
    assert not np.array_equal(expected, cv2.resize(legacy.render_frames_rgb[0].numpy(), (320, 180)))


def test_nonpositive_render_width_fails_before_opening_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="render_width"):
        load_clip_sequence(
            store_dir=tmp_path / "missing", clip_id="missing", sequence_length=2,
            image_size_hw=(64, 128), max_frames=2, normalize_imagenet=False,
            imagenet_mean=(0.0, 0.0, 0.0), imagenet_std=(1.0, 1.0, 1.0), render_width=0,
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_interval_preserves_frame_order_and_preprocessing(tmp_path: Path, normalize: bool) -> None:
    clip_id = "tracknet/game1/Clip1"
    store_dir = write_store_clip(tmp_path / "store", clip_id, [frame(i) for i in range(6)])

    def load(start: int, count: int | None) -> ClipSequence:
        return load_clip_sequence(
            store_dir=store_dir, clip_id=clip_id, sequence_length=2,
            image_size_hw=(32, 64), max_frames=count, normalize_imagenet=normalize,
            imagenet_mean=(0.485, 0.456, 0.406), imagenet_std=(0.229, 0.224, 0.225),
            render_width=80, start_frame=start,
        )

    full = load(0, None)
    interval = load(2, 3)
    assert interval.frame_names == tuple(f"{clip_id}:{i}" for i in range(2, 5))
    torch.testing.assert_close(interval.model_images, full.model_images[2:5], rtol=0, atol=0)
    for actual, expected in zip(interval.render_frames_rgb, full.render_frames_rgb[2:5], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("start,count,message", [(-1, 2, "nonnegative"), (4, 2, "outside"), (3, 1, "window")])
def test_unusable_intervals_are_rejected(
    tmp_path: Path, start: int, count: int, message: str,
) -> None:
    clip_id = "tracknet/game1/Clip1"
    store_dir = write_store_clip(tmp_path / "store", clip_id, [frame(i) for i in range(4)])
    with pytest.raises(ValueError, match=message):
        load_clip_sequence(
            store_dir=store_dir, clip_id=clip_id, sequence_length=2,
            image_size_hw=(32, 64), max_frames=count, normalize_imagenet=False,
            imagenet_mean=(0.0, 0.0, 0.0), imagenet_std=(1.0, 1.0, 1.0), start_frame=start,
        )
