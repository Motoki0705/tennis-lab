from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.runtime.contracts import Annotation


def test_request_uses_valid_short_example_and_only_two_project_texts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project_kits"
    contents, kit_id = build_kit(root)
    other, other_id = build_kit(tmp_path / "other" / "project_kits")
    assert kit_id == other_id
    assert contents == other
    assert {path.name for path in root.iterdir()} == {
        "PROJECT_INSTRUCTIONS.txt",
        "REQUEST.txt",
    }
    assert all(path.is_file() for path in root.iterdir())
    request = (root / "REQUEST.txt").read_text(encoding="utf-8")
    example = re.search(r"```json\n(.*?)\n```", request, re.S)
    assert example is not None
    Annotation.model_validate(json.loads(example.group(1)))
    for name in (
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    ):
        assert name not in request
    assert "court_definition.json" not in contents
    assert contents["PROTOCOL.md"].decode() in request
    assert len(request) < 4000
    mtimes = {path: path.stat().st_mtime_ns for path in root.iterdir()}
    assert build_kit(root) == (contents, kit_id)
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
        build_kit(tmp_path / "project_kits")
