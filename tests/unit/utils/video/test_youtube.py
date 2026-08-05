from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.video.youtube import find_downloaded_video, h264_encoder_args


def test_find_downloaded_video_ignores_metadata_and_partial_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "abc.info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "abc.part").write_text("", encoding="utf-8")
    video = tmp_path / "abc.mkv"
    video.write_text("video", encoding="utf-8")

    assert find_downloaded_video(tmp_path, "abc") == video


def test_h264_encoder_args_libx264() -> None:
    assert h264_encoder_args(
        encoder="libx264",
        preset="veryfast",
        pix_fmt="yuv420p",
        crf=24,
    ) == [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
    ]


def test_h264_encoder_args_nvenc_explicit_values() -> None:
    args = h264_encoder_args(
        encoder="h264_nvenc",
        preset="p5",
        pix_fmt="yuv420p",
        crf=20,
        tune="hq",
        rate_control="vbr",
        cq=20,
        bitrate="0",
        profile="high",
    )

    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert args[-2:] == ["-profile:v", "high"]


@pytest.mark.parametrize("missing", ["tune", "rate_control", "cq", "bitrate"])
def test_h264_encoder_args_nvenc_rejects_implicit_defaults(missing: str) -> None:
    values: dict[str, str | int | None] = {
        "tune": "hq",
        "rate_control": "vbr",
        "cq": 20,
        "bitrate": "0",
    }
    values[missing] = None

    with pytest.raises(ValueError, match=rf"missing .*{missing}"):
        h264_encoder_args(
            encoder="h264_nvenc",
            preset="p5",
            pix_fmt="yuv420p",
            crf=20,
            tune=values["tune"],  # type: ignore[arg-type]
            rate_control=values["rate_control"],  # type: ignore[arg-type]
            cq=values["cq"],  # type: ignore[arg-type]
            bitrate=values["bitrate"],  # type: ignore[arg-type]
        )
