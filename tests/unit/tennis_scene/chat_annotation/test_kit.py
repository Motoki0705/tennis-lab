from __future__ import annotations

from pathlib import Path

from src.tennis_scene.chat_annotation.kit import build_kit, write_clip_kit
from src.tennis_scene.chat_annotation.runtime.contracts import read_json, sha256_file


def test_kit_is_deterministic_and_only_project_texts_are_stored(tmp_path: Path) -> None:
    root = tmp_path / "project_kits"
    contents, kit_id = build_kit(root)
    other, other_id = build_kit(tmp_path / "other_project_kits")
    assert kit_id == other_id
    assert contents == other
    assert set(contents) == {
        "PROTOCOL.md",
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    }
    assert {path.name for path in root.iterdir()} == {
        "PROJECT_INSTRUCTIONS.txt",
        "REQUEST.txt",
    }
    assert all(path.is_file() for path in root.iterdir())
    mtimes = {path: path.stat().st_mtime_ns for path in root.iterdir()}
    assert build_kit(root) == (contents, kit_id)
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}


def test_attachments_are_written_directly_to_each_clip(tmp_path: Path) -> None:
    contents, kit_id = build_kit(tmp_path / "project_kits")
    for name in ("clip_a", "clip_b"):
        directory = tmp_path / name
        directory.mkdir()
        write_clip_kit(contents, directory)
        assert {path.name for path in directory.iterdir()} == set(contents)
        assert all(
            path.is_file() and not path.is_symlink() for path in directory.iterdir()
        )
        manifest = read_json(directory / "kit_manifest.json")
        assert manifest["kit_id"] == kit_id
        assert set(manifest["files"]) == set(contents) - {"kit_manifest.json"}
        for filename, digest in manifest["files"].items():
            assert sha256_file(directory / filename) == digest
