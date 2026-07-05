"""Behavioral tests for the 2026-07 utils-extraction campaign.

Each test pins the behavior of a helper that was consolidated into
``src/utils`` / ``src/tasks/base`` from duplicated task-local copies, and
(where cheap) that the old import paths still resolve to the shared
implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.utils.data.augmentation import tensor_images_to_uint8_rgb
from src.utils.data.heatmaps import resize_heatmap_sequence
from src.utils.geometry.image_size import resize_short_side_aligned
from src.utils.io import find_existing_file, load_json, save_json
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    net_height_at_x,
)
from src.utils.tensor_utils import flatten_time_to_batch, restore_time_from_batch
from src.utils.video.windows import build_window_starts, chunked


class TestNetHeightAtX:
    def test_center_and_post_heights(self) -> None:
        assert net_height_at_x(0.0) == pytest.approx(NET_HEIGHT_CENTER)
        assert net_height_at_x(HALF_DOUBLES_WIDTH) == pytest.approx(NET_HEIGHT_POST)
        assert net_height_at_x(-HALF_DOUBLES_WIDTH) == pytest.approx(NET_HEIGHT_POST)

    def test_clamped_beyond_posts(self) -> None:
        assert net_height_at_x(99.0) == pytest.approx(NET_HEIGHT_POST)

    def test_matches_replaced_formula(self) -> None:
        for x in (-7.0, -2.5, 0.0, 1.234, 5.485, 10.0):
            x_ratio = min(abs(x) / HALF_DOUBLES_WIDTH, 1.0)
            expected = NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER)
            assert net_height_at_x(x) == pytest.approx(expected)

    def test_old_call_sites_delegate(self) -> None:
        from src.tasks.blcs.generate_dataset.api_server import metrics

        assert metrics.net_height_at_x is net_height_at_x


class TestFindExistingFile:
    def test_first_matching_extension_wins(self, tmp_path) -> None:
        (tmp_path / "img.png").write_bytes(b"png")
        (tmp_path / "img.jpg").write_bytes(b"jpg")
        assert find_existing_file(tmp_path, "img", (".png", ".jpg")) == tmp_path / "img.png"
        assert find_existing_file(tmp_path, "img", (".jpg", ".png")) == tmp_path / "img.jpg"

    def test_fallback_extension(self, tmp_path) -> None:
        (tmp_path / "img.jpg").write_bytes(b"jpg")
        assert find_existing_file(tmp_path, "img", (".png", ".jpg")) == tmp_path / "img.jpg"

    def test_missing_returns_none(self, tmp_path) -> None:
        assert find_existing_file(tmp_path, "img", (".png", ".jpg")) is None


class TestResizeShortSideAligned:
    @pytest.mark.parametrize(
        ("width", "height", "short_side"),
        [(1920, 1080, 640), (1080, 1920, 640), (640, 640, 512), (1280, 720, 480)],
    )
    def test_matches_replaced_arithmetic(self, width: int, height: int, short_side: int) -> None:
        if height <= width:
            new_h = short_side
            new_w = int(round(width * new_h / height))
        else:
            new_w = short_side
            new_h = int(round(height * new_w / width))
        expected = ((new_w // 8) * 8, (new_h // 8) * 8)
        assert resize_short_side_aligned(width, height, short_side) == expected

    def test_align_parameter(self) -> None:
        assert resize_short_side_aligned(101, 101, 101, align=1) == (101, 101)
        assert resize_short_side_aligned(101, 101, 101, align=16) == (96, 96)


class TestTensorImagesToUint8Rgb:
    def test_matches_replaced_clamp_permute_cast(self) -> None:
        torch.manual_seed(0)
        frames = torch.rand(4, 3, 8, 9) * 1.4 - 0.2  # values outside [0, 1]
        clamped = frames.clamp(0.0, 1.0)
        expected = [
            (clamped[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8) for i in range(4)
        ]
        actual = tensor_images_to_uint8_rgb(frames)
        assert actual.shape == (4, 8, 9, 3)
        assert actual.dtype == np.uint8
        for exp, act in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(exp, act)

    def test_rejects_non_rgb_input(self) -> None:
        with pytest.raises(ValueError, match=r"\(\.\.\., 3, H, W\)"):
            tensor_images_to_uint8_rgb(torch.rand(4, 1, 8, 9))


class TestResizeHeatmapSequence:
    def test_matches_replaced_interpolate_block(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 32, 40)
        expected = (
            F.interpolate(
                logits.reshape(6, 1, 32, 40),
                size=(64, 80),
                mode="bilinear",
                align_corners=False,
            ).reshape(2, 3, 64, 80)
        )
        assert torch.equal(resize_heatmap_sequence(logits, (64, 80)), expected)

    def test_noop_when_size_matches(self) -> None:
        logits = torch.randn(2, 3, 32, 40)
        assert resize_heatmap_sequence(logits, (32, 40)) is logits

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, T, H, W\)"):
            resize_heatmap_sequence(torch.randn(2, 3, 4, 5, 6), (4, 5))


class TestWindowHelpers:
    def test_final_window_anchored(self) -> None:
        assert build_window_starts(frame_count=10, sequence_length=4, stride=3) == [0, 3, 6]
        assert build_window_starts(frame_count=11, sequence_length=4, stride=3) == [0, 3, 6, 7]

    def test_single_window(self) -> None:
        assert build_window_starts(frame_count=4, sequence_length=4, stride=2) == [0]

    def test_too_few_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="frame_count"):
            build_window_starts(frame_count=3, sequence_length=4, stride=1)

    def test_chunked(self) -> None:
        assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
        assert list(chunked([], 2)) == []

    def test_old_import_paths_resolve_to_shared(self) -> None:
        from src.tasks.ball_detection.visualization.adapters import (
            predict_inputs,
        )

        assert predict_inputs.build_window_starts is build_window_starts
        assert predict_inputs.chunked is chunked


class TestFlattenRestoreTime:
    def test_roundtrip(self) -> None:
        x = torch.randn(2, 5, 3, 4, 6)
        flat, batch_size, timesteps = flatten_time_to_batch(x)
        assert flat.shape == (6, 5, 4, 6)
        assert (batch_size, timesteps) == (2, 3)
        assert torch.equal(restore_time_from_batch(flat, batch_size, timesteps), x)

    def test_frame_ordering(self) -> None:
        # (B=1, C=1, T=2): frame t must land at flat index t.
        x = torch.arange(2.0).reshape(1, 1, 2, 1, 1)
        flat, _, _ = flatten_time_to_batch(x)
        assert flat[0].item() == 0.0
        assert flat[1].item() == 1.0


class TestConfigParsers:
    def test_parse_rgb_hw_float_triplet(self) -> None:
        from src.tasks.base.visualization.orchestrator import (
            parse_float_triplet,
            parse_hw,
            parse_rgb,
        )

        assert parse_rgb([1, 2, 3], name="c") == (1, 2, 3)
        assert parse_hw((288, 512), name="s") == (288, 512)
        assert parse_float_triplet([0.1, 0.2, 0.3], name="m") == (0.1, 0.2, 0.3)
        with pytest.raises(ValueError, match="length-3 RGB"):
            parse_rgb("abc", name="c")
        with pytest.raises(ValueError, match="length-2"):
            parse_hw([1, 2, 3], name="s")
        with pytest.raises(ValueError, match="length-3"):
            parse_float_triplet([1.0], name="m")


class TestPreviewHelpers:
    def test_draw_normalized_point_draws_in_place(self) -> None:
        from src.tasks.base.preview import draw_normalized_point

        image = np.zeros((21, 31, 3), dtype=np.uint8)
        draw_normalized_point(image, (0.5, 0.5), radius=2, color=(255, 0, 0), thickness=-1)
        assert image[10, 15].tolist() == [255, 0, 0]

    def test_compose_titled_row_layout(self) -> None:
        from src.tasks.base.preview import compose_titled_row

        cfg = OmegaConf.create(
            {
                "preview": {
                    "layout": {
                        "tile_gap": 4,
                        "header_height": 20,
                        "text_scale": 0.4,
                        "text_thickness": 1,
                        "background_rgb": [18, 18, 18],
                    }
                }
            }
        )
        panels = [np.zeros((10, 12, 3), dtype=np.uint8) for _ in range(3)]
        row = compose_titled_row(panels, ["a", "b", "c"], cfg)
        assert row.shape == (30, 3 * 12 + 2 * 4, 3)


class TestSaveJsonDefault:
    def test_default_serializer_passthrough(self, tmp_path) -> None:
        path = tmp_path / "meta.json"
        save_json({"path": tmp_path}, path, default=str)
        assert load_json(path) == {"path": str(tmp_path)}
