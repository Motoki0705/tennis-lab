from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.runtime.contracts import (
    BallAnnotation,
    PlayerAnnotation,
    Policies,
)


def test_request_embeds_target_schema_in_each_project_kit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project_kits"
    policies = Policies(ball_max_gap_seconds=0.1)
    contents, kit_id = build_kit(root, policies)
    other, other_id = build_kit(tmp_path / "other" / "project_kits", policies)
    assert kit_id == other_id
    assert contents == other
    assert {path.name for path in root.iterdir()} == {
        "ball_detection",
        "player_detection",
    }
    for target, model in (
        ("ball_detection", BallAnnotation),
        ("player_detection", PlayerAnnotation),
    ):
        directory = root / target
        assert {path.name for path in directory.iterdir()} == {
            "PROJECT_INSTRUCTIONS.txt",
            "REQUEST.txt",
        }
        request = (directory / "REQUEST.txt").read_text(encoding="utf-8")
        schema_match = re.search(r"```json\n(.*?)\n```", request, re.S)
        assert schema_match is not None
        schema = json.loads(schema_match.group(1))
        assert schema == model.model_json_schema()
        assert "動画1本につき1つの注釈JSON" in request
        assert "1つのZIPにまとめて提示" in request
        assert f"tennis_chat_{target.removesuffix('_detection')}_annotation.v1" in request
        assert len(request) > 3000
    assert "court_definition.json" not in contents
    mtimes = {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    assert build_kit(root, policies) == (contents, kit_id)
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}


@pytest.mark.parametrize("invalid_layout", ["flat", "nested", "orphan"])
def test_request_rejects_invalid_layout_and_unpublished_video(
    tmp_path: Path, invalid_layout: str
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    if invalid_layout == "flat":
        (videos / "orphan.mp4").write_bytes(b"unpublished")
    else:
        source = videos / "source"
        source.mkdir()
        if invalid_layout == "nested":
            (source / "clips").mkdir()
        else:
            (source / "orphan.mp4").write_bytes(b"unpublished")
    with pytest.raises(ValueError, match="directories only|only MP4|incomplete"):
        build_kit(tmp_path / "project_kits", Policies(ball_max_gap_seconds=0.1))
