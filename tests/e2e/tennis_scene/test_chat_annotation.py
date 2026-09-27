"""Prepare self-contained attachments and exercise the local reference runtime."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    PlayerAnnotation,
    make_template,
    read_json,
    write_json,
)
from src.tennis_scene.chat_annotation.runtime.media import probe_video


@pytest.mark.parametrize("vfr", [False, True])
def test_preparation_and_self_contained_clip(tmp_path: Path, vfr: bool) -> None:
    source = tmp_path / "fixture.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for index in range(155):
            image: NDArray[np.uint8] = np.zeros((360, 640, 3), dtype=np.uint8)
            image[40:300, 40:110] = (100, 210, 220)
            image[170:175, 300:305] = (230, 230, 0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index + (index // 5 if vfr else 0)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    repository = Path(__file__).parents[3]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tennis_scene.chat_annotation.scripts.prepare",
            f"paths.data_root={tmp_path}",
            f"paths.output_root={tmp_path / 'output'}",
            "source.local_video=fixture.mp4",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    summary_path = next((tmp_path / "output").rglob("prepared.json"))
    summary = read_json(summary_path)
    manifests = [
        ClipManifest.model_validate(
            read_json(summary_path.parent / "clips" / name / "clip_manifest.json")
        )
        for name in summary["clips"]
    ]
    if not vfr:
        assert [m.target_range.stop - m.target_range.start for m in manifests] == [
            65,
            65,
            25,
        ]
        assert [len(m.frames) for m in manifests] == [70, 75, 30]
    assert [
        f.source_frame_index for m in manifests for f in m.frames if f.is_target
    ] == list(range(155))
    project_texts = tmp_path / "output" / "chat_annotation" / "project_kits"
    assert {path.name for path in project_texts.iterdir()} == {
        "ball_detection",
        "player_detection",
    }
    assert "project_kit_directory" not in summary
    directory = tmp_path / "chat_upload"
    directory.mkdir()
    videos_root = project_texts.parent / "videos"
    assert {p.name for p in videos_root.iterdir()} == {
        Path(manifests[0].source.filename).stem
    }
    videos = videos_root / Path(manifests[0].source.filename).stem
    assert all(p.is_file() and p.suffix == ".mp4" for p in videos.iterdir())
    assert len(list(videos.iterdir())) == 3
    video = directory / manifests[0].filename
    shutil.copyfile(videos / video.name, video)
    request = (project_texts / "player_detection" / "REQUEST.txt").read_text(
        encoding="utf-8"
    )
    (directory / "REQUEST.txt").write_text(request, encoding="utf-8")
    assert {p.name for p in directory.iterdir()} == {video.name, "REQUEST.txt"}
    # A Chat can construct the result from its video and the concise request alone.
    # Source mapping remains local for validation after receiving that result.
    schema_match = re.search(r"```json\n(.*?)\n```", request, re.S)
    assert schema_match is not None
    assert "入力一覧" not in request and video.name not in request
    schema = json.loads(schema_match.group(1))
    assert "players" in schema["$defs"]["PlayerFrameAnnotation"]["properties"]
    assert "balls" not in schema["$defs"]["PlayerFrameAnnotation"]["properties"]
    timeline = probe_video(video)
    frame_count = len(timeline.pts)
    annotation = make_template(manifests[0], target="player").model_dump(
        mode="json"
    )
    annotation["status"] = "completed"
    for row in annotation["frames"]:
        row["reviewed"] = True
        row["players"] = [
            {
                "track_id": "p1",
                "bbox_xyxy": [40, 40, 110, 300],
                "bbox_source": "observed",
                "occluded": False,
                "truncated": False,
            }
        ]
    PlayerAnnotation.model_validate(annotation)
    annotation_path = directory / f"annotation_{video.stem}.json"
    write_json(annotation_path, annotation)
    manifest_path = (
        summary_path.parent / "clips" / manifests[0].clip_id / "clip_manifest.json"
    )

    def run(*arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.tennis_scene.chat_annotation.scripts.annotate",
                *arguments,
            ],
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    common = ["--manifest", str(manifest_path)]
    run("preflight", *common, "--video", str(video))
    template_path = tmp_path / "template.json"
    run(
        "init",
        *common,
        "--video",
        str(video),
        "--output",
        str(template_path),
        "--target",
        "player",
    )
    template = read_json(template_path)
    assert len(template["frames"]) == frame_count
    assert all(not row["reviewed"] for row in template["frames"])
    run(
        "frames",
        *common,
        "--video",
        str(video),
        "--output",
        str(tmp_path / "frames"),
        "--start",
        "0",
        "--stop",
        "2",
        "--crop",
        "280",
        "150",
        "340",
        "200",
    )
    assert len(list((tmp_path / "frames").glob("*.png"))) == 2
    crop_image = cv2.imread(str(tmp_path / "frames" / "frame_0000000_crop.png"))
    assert crop_image is not None
    with av.open(str(video)) as input_container:
        original = next(input_container.decode(video=0)).to_ndarray(format="bgr24")
    np.testing.assert_array_equal(crop_image[:50, :60], original[150:200, 280:340])

    def package(output: Path, expected_status: str) -> None:
        response = run(
            "finalize",
            *common,
            "--video",
            str(video),
            "--annotations",
            str(annotation_path),
            "--output",
            str(output),
        )
        assert expected_status in response.stdout
        archive = output / f"{video.stem}.zip"
        expected = {f"overlay_{video.stem}.mp4", annotation_path.name}
        assert {p.name for p in output.iterdir()} == expected | {archive.name}
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            assert set(bundle.namelist()) == expected
            assert bundle.read(annotation_path.name) == annotation_path.read_bytes()
        overlay = probe_video(output / f"overlay_{video.stem}.mp4")
        assert len(overlay.pts) == frame_count
        assert [p * overlay.time_base for p in overlay.pts] == [
            p * timeline.time_base for p in timeline.pts
        ]
        assert [d * overlay.time_base for d in overlay.durations] == [
            d * timeline.time_base for d in timeline.durations
        ]
        with av.open(str(output / f"overlay_{video.stem}.mp4")) as container:
            decoded = list(container.decode(video=0))
        # The formerly context-only last frame also has the JSON's cyan player box.
        pixels = decoded[-1].to_ndarray(format="bgr24")
        edge = pixels[39:42, 45:100]
        assert (
            np.count_nonzero(
                (edge[:, :, 0] > 150) & (edge[:, :, 1] > 150) & (edge[:, :, 2] < 80)
            )
            > 10
        )

    package(tmp_path / "completed", "completed")
    annotation["status"] = "partial"
    annotation["frames"][-1]["reviewed"] = False
    annotation["frames"][-1]["notes"] = "Frame not yet inspected."
    write_json(annotation_path, annotation)
    package(tmp_path / "partial", "partial")

    # Claiming completion for unreviewed frames cannot publish a success archive.
    annotation["status"] = "completed"
    write_json(annotation_path, annotation)
    failed = tmp_path / "failed"
    result = run(
        "finalize",
        *common,
        "--video",
        str(video),
        "--annotations",
        str(annotation_path),
        "--output",
        str(failed),
        success=False,
    )
    assert "failed" in result.stdout and not failed.exists()
    # Malformed JSON is not converted to a plausible empty overlay.
    annotation_path.write_text(
        json.dumps({"schema_version": annotation["schema_version"], "bad": True}),
        encoding="utf-8",
    )
    result = run(
        "finalize",
        *common,
        "--video",
        str(video),
        "--annotations",
        str(annotation_path),
        "--output",
        str(failed),
        success=False,
    )
    assert "ValidationError" in result.stdout and not failed.exists()
