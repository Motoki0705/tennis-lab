from __future__ import annotations

from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.runtime.contracts import read_json, sha256_file


def test_code_free_kit_is_deterministic_and_every_reference_is_attached(
    tmp_path: Path,
) -> None:
    directory, kit_id = build_kit(tmp_path / "first")
    other, other_id = build_kit(tmp_path / "second")
    assert kit_id == other_id
    expected = {
        "PROTOCOL.md",
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    }
    assert {path.name for path in directory.iterdir()} == expected
    for name in expected:
        assert (directory / name).read_bytes() == (other / name).read_bytes()
    manifest = read_json(directory / "kit_manifest.json")
    assert set(manifest["files"]) == expected - {"kit_manifest.json"}
    assert manifest["kit_id"] == kit_id
    for name, digest in manifest["files"].items():
        assert sha256_file(directory / name) == digest
    assert (directory.parent / "PROJECT_INSTRUCTIONS.txt").is_file()
    assert (directory.parent / "PROTOCOL.md").read_bytes() == (
        directory / "PROTOCOL.md"
    ).read_bytes()
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in directory.parent.rglob("*")
        if path.is_file()
    }
    assert build_kit(directory.parent) == (directory, kit_id)
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}


@pytest.mark.parametrize("modification", ["extra", "changed", "missing"])
def test_existing_kit_must_not_be_silently_replaced(
    tmp_path: Path, modification: str
) -> None:
    directory, _ = build_kit(tmp_path)
    if modification == "extra":
        (directory / "annotation_tools.py").write_text(
            "legacy helper", encoding="utf-8"
        )
    elif modification == "changed":
        (directory / "PROTOCOL.md").write_text("changed", encoding="utf-8")
    else:
        (directory / "annotation.schema.json").unlink()
    with pytest.raises(ValueError, match="modified"):
        build_kit(tmp_path)
