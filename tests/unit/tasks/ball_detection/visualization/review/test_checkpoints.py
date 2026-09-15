"""Unit tests for ball-detection checkpoint discovery and metadata."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_detection.visualization.review.checkpoints import (
    DEFAULT_NMS_KERNEL,
    DEFAULT_PEAK_THRESHOLD,
    MINIMUM_FRAMES_BY_MODEL,
    checkpoint_roots,
    describe_checkpoint,
    describe_config,
    read_metrics_settings,
    scan_checkpoints,
)


def test_catalog_minimums_match_model_io_factory() -> None:
    """The catalog mirrors the factory's per-architecture frame minimums."""
    assert set(MINIMUM_FRAMES_BY_MODEL) == {"stunet", "conv_next_unet", "dinov3_rope"}
    for model_name, minimum in (("stunet", 8), ("conv_next_unet", 1)):
        model: dict[str, object] = {
            "name": model_name,
            "input_mode": "mdd",
            "in_channels": 2,
            "num_classes": 1,
            "num_frames": minimum,
            "input_layout": "bcthw",
            "mdd_a": 0.2,
            "mdd_b": 0.15,
        }
        if model_name == "conv_next_unet":
            model.update({"dims": [4, 8, 16, 32], "depth": 1, "drop_path_prob": 0.0})
        bound = build_ball_detection_pair(OmegaConf.create({"model": model}))
        assert bound.adapter.minimum_frames == minimum
        assert MINIMUM_FRAMES_BY_MODEL[model_name] == minimum


def test_missing_metrics_block_uses_documented_defaults_with_warning() -> None:
    settings = read_metrics_settings({})
    assert settings.errors == ()
    assert settings.defaults.peak_threshold == DEFAULT_PEAK_THRESHOLD
    assert settings.defaults.nms_kernel == DEFAULT_NMS_KERNEL
    assert settings.defaults.subpixel_refine is True
    assert any("no saved metrics block" in text for text in settings.warnings)


def test_missing_metrics_keys_are_warned_but_kept_usable() -> None:
    settings = read_metrics_settings({"metrics": {"peak_threshold": 0.35}})
    assert settings.errors == ()
    assert settings.defaults.peak_threshold == 0.35
    assert settings.defaults.nms_kernel == DEFAULT_NMS_KERNEL
    assert any("nms_kernel" in text for text in settings.warnings)


def test_invalid_metrics_values_are_errors_not_silent_defaults() -> None:
    settings = read_metrics_settings(
        {
            "metrics": {
                "peak_threshold": 1.4,
                "ball_distance_threshold": 0.0,
                "nms_kernel": 4,
                "max_predictions_per_frame": -1,
                "subpixel_refine": "yes",
            }
        }
    )
    assert len(settings.errors) == 5
    assert any("peak_threshold" in text for text in settings.errors)
    assert any("nms_kernel" in text for text in settings.errors)
    assert any("subpixel_refine" in text for text in settings.errors)


def test_non_finite_metrics_values_are_errors() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        settings = read_metrics_settings({"metrics": {"peak_threshold": value}})
        assert settings.errors, f"{value!r} must not be accepted"
        nan_distance = read_metrics_settings(
            {"metrics": {"ball_distance_threshold": value}}
        )
        assert nan_distance.errors


def test_valid_metrics_block_is_read_verbatim() -> None:
    settings = read_metrics_settings(
        {
            "metrics": {
                "peak_threshold": 0.35,
                "ball_distance_threshold": 6.0,
                "nms_kernel": 5,
                "max_predictions_per_frame": 3,
                "subpixel_refine": False,
            }
        }
    )
    assert settings.errors == ()
    assert settings.warnings == ()
    assert settings.defaults.peak_threshold == 0.35
    assert settings.defaults.ball_distance_threshold == 6.0
    assert settings.defaults.nms_kernel == 5
    assert settings.defaults.max_predictions_per_frame == 3
    assert settings.defaults.subpixel_refine is False


def test_describe_config_carries_architecture_contract() -> None:
    info = describe_config(
        checkpoint_id="run.ckpt",
        path=Path("/tmp/run.ckpt"),
        label="run",
        config=OmegaConf.create(
            {
                "model": {
                    "name": "conv_next_unet",
                    "input_mode": "mdd",
                    "num_frames": 8,
                },
                "data": {"image_size": [288, 512]},
            }
        ),
    )
    assert info.usable
    assert info.model == "conv_next_unet"
    assert info.num_frames == 8
    assert info.minimum_window == 2
    assert info.maximum_window == 8
    assert info.image_size_hw == (288, 512)
    payload = info.to_dict(compatible_datasets=["tracknet"])
    assert payload["settings"] == {"count": 8, "threshold": DEFAULT_PEAK_THRESHOLD}
    assert payload["window"] == {"min": 2, "max": 8}


def test_describe_config_rejects_short_temporal_window() -> None:
    info = describe_config(
        checkpoint_id="short.ckpt",
        path=Path("/tmp/short.ckpt"),
        label="short",
        config=OmegaConf.create(
            {"model": {"name": "stunet", "input_mode": "mdd", "num_frames": 4}}
        ),
    )
    assert not info.usable
    assert info.error is not None and "at least 8 frames" in info.error
    assert MINIMUM_FRAMES_BY_MODEL["stunet"] == 8


def test_describe_config_rejects_unknown_model() -> None:
    info = describe_config(
        checkpoint_id="other.ckpt",
        path=Path("/tmp/other.ckpt"),
        label="other",
        config=OmegaConf.create(
            {"model": {"name": "mystery", "input_mode": "rgb", "num_frames": 2}}
        ),
    )
    assert not info.usable
    assert info.error is not None and "unsupported model.name" in info.error


def test_describe_config_without_model_block() -> None:
    info = describe_config(
        checkpoint_id="empty.ckpt",
        path=Path("/tmp/empty.ckpt"),
        label="empty",
        config=OmegaConf.create({}),
    )
    assert not info.usable
    assert info.error is not None and "model.name" in info.error


def test_describe_checkpoint_reports_unreadable_body(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.ckpt"
    corrupt.write_text("this is not a torch checkpoint", encoding="utf-8")
    info = describe_checkpoint(
        checkpoint_id="corrupt.ckpt", path=corrupt, label="corrupt"
    )
    assert not info.usable
    assert info.error is not None


def test_scan_checkpoints_prefixes_curated_root(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    output_root = tmp_path / "outputs"
    checkpoint_root = tmp_path / "ckpt"
    make_tiny_checkpoint(output_root / "run-local.ckpt")
    make_tiny_checkpoint(checkpoint_root / "ball_detection" / "run-i618.ckpt")

    roots = checkpoint_roots(output_root, checkpoint_root / "ball_detection")
    assert roots[0] == (output_root, "")
    assert roots[1][1] == "ckpt/ball_detection/"

    infos = scan_checkpoints(roots)
    ids = [info.id for info in infos]
    assert ids == ["ckpt/ball_detection/run-i618.ckpt", "run-local.ckpt"]
    assert all(info.usable for info in infos)


def test_scan_checkpoints_does_not_duplicate_nested_roots(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    outputs = tmp_path / "outputs" / "ball_detection"
    make_tiny_checkpoint(outputs / "run-local.ckpt")
    roots = checkpoint_roots(tmp_path / "outputs", outputs)
    # The curated root is nested inside the outputs root; it is added once, with
    # a prefix, and the same file is never described twice.
    assert [root for root, _ in roots] == [
        tmp_path / "outputs",
        tmp_path / "outputs" / "ball_detection",
    ]
    # The primary root's recursive scan reaches the nested file first, so the id
    # carries the nested relative path and the file is described exactly once.
    assert [info.id for info in scan_checkpoints(roots)] == [
        "ball_detection/run-local.ckpt"
    ]


def test_scan_checkpoints_deduplicates_identical_ids(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    make_tiny_checkpoint(first / "run.ckpt")
    make_tiny_checkpoint(second / "run.ckpt")
    infos = scan_checkpoints(((first, ""), (second, "")))
    ids = sorted(info.id for info in infos)
    assert ids == ["a/run.ckpt", "b/run.ckpt"]


def test_scan_checkpoints_ignores_missing_roots(tmp_path: Path) -> None:
    assert scan_checkpoints(((tmp_path / "absent", ""),)) == []


def test_scan_checkpoints_skips_directories(tmp_path: Path) -> None:
    """A directory named ``*.ckpt`` is skipped rather than reported as usable."""
    directory = tmp_path / "outputs" / "weird.ckpt"
    directory.mkdir(parents=True)
    assert scan_checkpoints(((tmp_path / "outputs", ""),)) == []


@pytest.mark.parametrize("model_name", ["stunet", "conv_next_unet", "dinov3_rope"])
def test_minimum_frames_mapping_covers_every_factory_model(model_name: str) -> None:
    assert model_name in MINIMUM_FRAMES_BY_MODEL
