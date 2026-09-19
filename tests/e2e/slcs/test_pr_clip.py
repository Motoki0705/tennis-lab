"""CPU-only composition contract, including differently sampled source streams."""

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.visualization.slcs_pr_clip import (
    Reader,
    Stream,
    parser,
    probe,
    render,
    sample_times,
)


def test_time_correspondence_and_duration_validation() -> None:
    rgb = Stream("rgb", 60, 60, 16, 16)
    scene = Stream("scene", 20, 20, 16, 16)
    assert sample_times(rgb, scene, 0.1, 0.9, 10) == pytest.approx(
        [0.1 + i / 10 for i in range(8)]
    )
    with pytest.raises(ValueError, match="last-sample"):
        sample_times(rgb, Stream("short", 20, 17, 16, 16), 0, 0.5, 10)
    with pytest.raises(ValueError, match="duration"):
        sample_times(rgb, scene, 0, 1.1, 10)
    with pytest.raises(ValueError, match="last frame"):
        sample_times(rgb, scene, 0.98, 1, 10)
    with pytest.raises(ValueError, match="Require"):
        sample_times(rgb, scene, float("nan"), 1, 10)


def test_missing_and_invalid_video(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing"):
        probe(tmp_path / "absent.mp4")
    invalid = tmp_path / "bad.mp4"
    invalid.write_bytes(b"not video")
    with pytest.raises(subprocess.CalledProcessError):
        probe(invalid)


def test_actual_video_composition_and_provenance(tmp_path: Path) -> None:
    for name, rate, color in (("rgb", 12, "red"), ("scene", 4, "blue")):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x32:r={rate}:d=1",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(tmp_path / f"{name}.mp4"),
            ],
            check=True,
        )
    for name, expected_index in (("rgb", 6), ("scene", 2)):
        reader = Reader(probe(tmp_path / f"{name}.mp4"))
        try:
            reader.at(0.5)
            assert reader.index == expected_index
            with pytest.raises(ValueError, match="sequential"):
                reader.at(0)
        finally:
            reader.close()
    args = parser().parse_args(
        [
            "--overlay",
            str(tmp_path / "rgb.mp4"),
            "--scene",
            str(tmp_path / "scene.mp4"),
            "--experiment",
            "example",
            "--run-id",
            "trial",
            "--label",
            "Fixture",
            "--model",
            "baseline",
            "--epoch",
            "56",
            "--clip-id",
            "video/clip",
            "--camera-id",
            "cam0",
            "--checkpoint-sha256",
            "a" * 64,
            "--start",
            "0",
            "--end",
            "0.75",
            "--fps",
            "4",
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )
    output = render(args)
    video = probe(output / "comparison.mp4")
    assert (video.frames, video.fps) == (3, 4)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["labels"]["epoch"] == 56
    assert provenance["labels"]["clip_id"] == "video/clip"
    assert provenance["labels"]["checkpoint_sha256"] == "a" * 64
    assert [s["fps"] for s in provenance["sources"]] == [12, 4]
    assert all(len(s["sha256"]) == 64 for s in provenance["sources"])
    assert provenance["contact_sheet_times_seconds"] == [0, 0.25, 0.5]
    assert "not measured" in provenance["teacher"]
    with Image.open(output / "contact_sheet.png") as sheet:
        assert sheet.size == (1440, 1830)
        # Full-frame left/right sources remain visibly red/blue after scaling.
        left, right = sheet.getpixel((350, 300)), sheet.getpixel((1050, 300))
        assert isinstance(left, tuple) and left[0] > 240
        assert isinstance(right, tuple) and right[2] > 240
    with pytest.raises(FileExistsError):
        render(args)
