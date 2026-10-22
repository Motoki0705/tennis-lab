"""Asset CLI path validation runs before packaging or verification side effects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.synthetic_data_generation.scripts import court_scene_assets
from src.utils.configuration.errors import PathContractError


@pytest.mark.parametrize("command", ["pack", "verify"])
def test_main_passes_validated_paths_and_pinned_versions(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "output" / "assets"
    expected = {"manifest_sha256": "c" * 64}

    def pack(source_root, output, *, scene_ids, repository_commit, nht_commit):
        assert source_root == source
        assert output == destination
        assert scene_ids == ["B00", "B01"]
        assert repository_commit == "a" * 40
        assert nht_commit == "b" * 40
        return expected

    def verify(root, *, manifest_sha256):
        assert root == source
        assert manifest_sha256 == "c" * 64
        return expected

    def git(argv, *, text):
        assert text is True
        if argv[1] == "status":
            return ""
        return "a" * 40 if argv[-1] == "HEAD" else "b" * 40

    monkeypatch.setattr(court_scene_assets, "pack_scene_assets", pack)
    monkeypatch.setattr(court_scene_assets, "verify_scene_assets", verify)
    monkeypatch.setattr(court_scene_assets.subprocess, "check_output", git)
    monkeypatch.chdir(tmp_path)
    options = (
        [
            "--source-root",
            "source",
            "--destination",
            "output/assets",
            "--scene-ids",
            "B00",
            "B01",
        ]
        if command == "pack"
        else ["--root", "source", "--manifest-sha256", "c" * 64]
    )
    monkeypatch.setattr(sys, "argv", ["court_scene_assets", command, *options])

    court_scene_assets.main()

    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize("command", ["pack", "verify"])
@pytest.mark.parametrize("input_kind", ["missing", "file", "symlink"])
def test_invalid_paths_fail_before_git_or_asset_operations(
    command: str,
    input_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "new" / "assets"
    if input_kind == "file":
        source.rmdir()
        source.write_text("not a directory")
    elif input_kind == "missing":
        source.rmdir()
    elif command == "pack":
        destination.parent.mkdir()
        destination.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        linked_source = tmp_path / "linked-source"
        linked_source.symlink_to(source, target_is_directory=True)
        source = linked_source

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid paths reached an asset operation or Git probe.")

    monkeypatch.setattr(court_scene_assets, "pack_scene_assets", unexpected)
    monkeypatch.setattr(court_scene_assets, "verify_scene_assets", unexpected)
    monkeypatch.setattr(court_scene_assets.subprocess, "check_output", unexpected)
    options = (
        [
            "--source-root",
            str(source),
            "--destination",
            str(destination),
            "--scene-ids",
            "B00",
        ]
        if command == "pack"
        else ["--root", str(source), "--manifest-sha256", "c" * 64]
    )
    monkeypatch.setattr(sys, "argv", ["court_scene_assets", command, *options])

    error = (
        FileExistsError
        if command == "pack" and input_kind == "symlink"
        else ValueError
        if input_kind == "symlink"
        else PathContractError
    )
    with pytest.raises(error):
        court_scene_assets.main()

    assert not destination.exists()
