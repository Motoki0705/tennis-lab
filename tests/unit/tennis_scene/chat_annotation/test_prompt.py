from __future__ import annotations

import json
import re
from pathlib import Path

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.prompt import render_request
from src.tennis_scene.chat_annotation.runtime.contracts import ClipManifest, FrameRange


def test_request_keeps_only_essential_metadata_for_all_clip_frames(
    tmp_path: Path,
    manifest: ClipManifest,
) -> None:
    manifest.target_range = FrameRange(start=1, stop=11)
    manifest.frames[0].is_target = manifest.frames[-1].is_target = False
    contents, _ = build_kit(tmp_path / "project_kits")
    request = render_request(contents, [manifest])
    block = re.search(r"```jsonl\n(.*?)\n```", request, re.S)
    assert block is not None
    record = json.loads(block.group(1))
    assert record == {
        "filename": "sample.mp4",
        "width": 1920,
        "height": 1080,
        "frame_count": 12,
        "ball_max_gap_seconds": 0.1,
    }
    assert "frame_runs" not in request
    assert "source_pts" not in request
    assert "court_definition" not in request


def test_five_clip_request_is_bounded_and_does_not_repeat_schema(
    tmp_path: Path,
    manifest: ClipManifest,
) -> None:
    contents, _ = build_kit(tmp_path / "project_kits")
    manifests = [
        manifest.model_copy(update={"filename": f"source__run__clip_{i}.mp4"})
        for i in range(5)
    ]
    request = render_request(contents, manifests)
    assert all(m.filename in request for m in manifests)
    assert len(request) < 5000
    assert "$defs" not in request
