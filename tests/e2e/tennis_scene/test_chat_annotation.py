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
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.prompt import expand_manifest
from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    read_json,
    sha256_file,
    write_json,
)
from src.tennis_scene.chat_annotation.runtime.media import probe_video


def test_preparation_and_self_contained_clip(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for index in range(155):
            image: NDArray[np.uint8] = np.zeros((360, 640, 3), dtype=np.uint8)
            image[40:300, 40:110] = (100, 210, 220)
            image[170:175, 300:305] = (230, 230, 0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
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
        "PROJECT_INSTRUCTIONS.txt",
        "REQUEST.txt",
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
    request = (project_texts / "REQUEST.txt").read_text(encoding="utf-8")
    (directory / "REQUEST.txt").write_text(request, encoding="utf-8")
    assert {p.name for p in directory.iterdir()} == {video.name, "REQUEST.txt"}
    # Only the video and the pasted text remain available as input data.
    project_texts.parent.rename(tmp_path / "unavailable_preparation")
    blocks = {
        name: json.loads(value)
        for name, value in re.findall(
            r"## ([^\n]+)\n\n```json\n(.*?)\n```", request, re.S
        )
    }
    record = next(
        value for value in blocks["動画入力定義"] if value["filename"] == video.name
    )
    recovered = expand_manifest(record)
    assert recovered == manifests[0]
    manifest_path = directory / "clip_manifest.json"
    write_json(manifest_path, recovered.model_dump(mode="json"))
    for name in (
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    ):
        write_json(directory / name, blocks[name])
    protocol = request[
        request.index("# テニス動画") : request.index("\n## annotation.schema.json")
    ]
    (directory / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    for name, digest in blocks["kit_manifest.json"]["files"].items():
        assert sha256_file(directory / name) == digest
    annotation_path = tmp_path / "annotations.json"

    def run(*arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.tennis_scene.chat_annotation.runtime.cli import main; "
                "from pathlib import Path; raise SystemExit(main(Path.cwd()))",
                "--kit-dir",
                str(directory),
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
    run("init", *common, "--video", str(video), "--output", str(annotation_path))
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
    label = "crop origin=(280,150) size=(60,50); return ORIGINAL pixel xy"
    assert (
        crop_image.shape[1]
        >= cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 8
    )
    with av.open(str(video)) as input_container:
        original = next(input_container.decode(video=0)).to_ndarray(format="bgr24")
    np.testing.assert_array_equal(crop_image[:50, :60], original[150:200, 280:340])
    annotation = read_json(annotation_path)
    annotation["inspection_ranges"] = [{"start": 0, "stop": 65}]
    annotation["court_mode"] = "unavailable"
    for row in annotation["frames"]:
        row["people_review"] = row["balls_review"] = row["court_review"] = "complete"
        row["people"] = [
            {
                "track_id": "p1",
                "kind": "player",
                "non_player_role": None,
                "court_relation": "target",
                "bbox_xyxy": [40, 40, 110, 300],
                "bbox_source": "observed",
                "occluded": False,
                "truncated": False,
                "source_frames": [row["frame_index"]],
            }
        ]
        row["balls"] = [
            {
                "track_id": "b1",
                "center_px": [302, 172],
                "status": "visible",
                "missing_reason": None,
                "source_frames": [row["frame_index"]],
            }
        ]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    output = tmp_path / "result"
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
    assert "状態: completed" in response.stdout
    assert len(response.stdout.strip().splitlines()) == 5
    archive = next(output.glob("*.zip"))
    assert f"sandbox:{archive}" in response.stdout
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert {
            "annotations.json",
            "clip_manifest.json",
            "overlay.mp4",
            "contact_sheet.jpg",
            "validation_report.json",
            "provenance.json",
            "kit_manifest.json",
            "FINAL_RESPONSE.txt",
        } <= set(bundle.namelist())
        assert bundle.read("clip_manifest.json") == manifest_path.read_bytes()
        assert bundle.read("annotations.json") == annotation_path.read_bytes()
        assert (
            json.loads(bundle.read("validation_report.json"))["reviewed_frames"] == 65
        )
    assert len(probe_video(output / "overlay.mp4").pts) == 70
    # Invalid annotations yield an auditable failure ZIP, never a plausible empty overlay.
    annotation_path.write_text('{"bad":true}', encoding="utf-8")
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
    assert "状態: failed" in result.stdout
    assert list(failed.glob("*.zip"))
    assert not (failed / "overlay.mp4").exists()
    # Pydantic errors include newlines; the final failure response still has five lines.
    bad_manifest = tmp_path / "bad_manifest.json"
    malformed = read_json(manifest_path)
    malformed["width"] = "invalid"
    bad_manifest.write_text(json.dumps(malformed), encoding="utf-8")
    invalid = run(
        "preflight",
        "--manifest",
        str(bad_manifest),
        "--video",
        str(video),
        success=False,
    )
    assert len(invalid.stdout.strip().splitlines()) == 5
    assert "ValidationError" in invalid.stdout
    invalid_annotation = run(
        "validate",
        *common,
        "--annotations",
        str(annotation_path),
        "--report",
        str(tmp_path / "invalid_report.json"),
        success=False,
    )
    assert len(invalid_annotation.stdout.strip().splitlines()) == 5
    # Corrupting an attached requirement is detected before local annotation/rendering.
    (directory / "PROTOCOL.md").write_text("modified", encoding="utf-8")
    rejected = run("preflight", *common, "--video", str(video), success=False)
    assert "missing or modified" in rejected.stdout
