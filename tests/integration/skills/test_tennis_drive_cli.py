"""Exercise actual rclone copy/hash/move against an isolated local alias."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def drive_cli(tmp_path: Path) -> tuple[Callable[..., dict[str, Any]], Path]:
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("rclone executable required for isolated local integration")
    assert rclone is not None
    executable: str = rclone
    remote = tmp_path / "remote"
    project = remote / "tennis_lab"
    project.mkdir(parents=True)
    config = tmp_path / "rclone.conf"
    config.write_text(f"[test]\ntype = alias\nremote = {remote}\n")
    script = (
        Path(__file__).resolve().parents[3]
        / ".agents/skills/tennis-drive/scripts/drive.py"
    )
    env = {**os.environ, "RCLONE_CONFIG": str(config)}

    def invoke(*args: str, expected: int = 0) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--remote-root",
                "test:tennis_lab",
                "--rclone-bin",
                executable,
                *args,
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        payload: dict[str, Any] = json.loads(result.stdout)
        assert payload["schema_version"] == 1
        assert payload["rclone_config"] == str(config.resolve())
        return payload

    return invoke, project


@pytest.fixture
def drive_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".agents/skills/tennis-drive/scripts"))
    return importlib.import_module("drive")


def test_session_reuses_one_process_and_copy_never_overwrites_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drive_module: ModuleType
) -> None:
    core = importlib.import_module("drive_core")
    executable = shutil.which("rclone")
    if executable is None:
        pytest.skip("rclone executable required for isolated session integration")
    config = tmp_path / "session.conf"
    config.write_text("[fixture]\ntype = local\n")
    backend = core.RcloneBackend(
        remote_root="fixture:project",
        executable=executable,
        config_path=config,
        timeout=10,
    )
    starts = []
    real_popen = core.subprocess.Popen

    def popen(*args, **kwargs):
        starts.append(args[0])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(core.subprocess, "Popen", popen)
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"frozen")
    with core.RcloneSession(backend) as session:
        process = session.process
        assert session.url.startswith("http://127.0.0.1:")
        params = {
            "srcFs": str(tmp_path),
            "srcRemote": source.name,
            "dstFs": str(tmp_path),
            "dstRemote": destination.name,
            "_config": {
                "Immutable": True,
                "IgnoreExisting": True,
                "IgnoreChecksum": True,
                "IgnoreSize": True,
            },
        }
        session.call("operations/copyfile", **params)
        session.call("operations/copyfile", **params)
        assert destination.read_bytes() == b"frozen"
        source.write_bytes(b"change")
        session.call("operations/copyfile", **params)
        assert destination.read_bytes() == b"frozen"
        with pytest.raises(core.DriveToolError, match="diagnostics withheld") as error:
            session.call("operations/stat", fs="not_configured:", remote="secret-name")
        assert "secret-name" not in str(error.value)
        assert session.process is process
    assert process.poll() is not None
    assert len(starts) == 1
    assert "127.0.0.1:0" in starts[0]
    with pytest.raises(core.DriveToolError, match="closed"):
        session.call("rc/noop")


def test_session_stops_its_process_when_caller_fails(
    tmp_path: Path, drive_module: ModuleType
) -> None:
    core = importlib.import_module("drive_core")
    executable = shutil.which("rclone")
    if executable is None:
        pytest.skip("rclone executable required for isolated session integration")
    config = tmp_path / "session.conf"
    config.write_text("[fixture]\ntype = local\n")
    backend = core.RcloneBackend(
        remote_root="fixture:project",
        executable=executable,
        config_path=config,
        timeout=10,
    )
    with (
        pytest.raises(RuntimeError, match="caller failed"),
        core.RcloneSession(backend) as session,
    ):
        process = session.process
        raise RuntimeError("caller failed")
    assert process.poll() is not None


def test_transfer_verify_relocate_and_inventory(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "input"
    source.mkdir()
    (source / "empty-dir").mkdir()
    (source / "quoted ' name.txt").write_text("tennis\n")
    planned = call("upload", str(source), "data/fixture-v1", "--dry-run", "--verify")
    assert planned["status"] == "planned" and not (remote / "data/fixture-v1").exists()
    uploaded = call("upload", str(source), "data/fixture-v1", "--verify")
    assert uploaded["verified"]
    assert (remote / "data/fixture-v1/empty-dir").is_dir()
    call("upload", str(source), "data/fixture-v1", expected=1)
    assert call("inspect", "data/fixture-v1")["file_count"] == 1
    assert call("verify", str(source), "data/fixture-v1")["matches"]
    copied = call("copy", "data/fixture-v1", "staging/copy")
    assert copied["verified"]
    assert (remote / "data/fixture-v1").exists()
    call("move", "staging/copy", "staging/renamed")
    assert not (remote / "staging/copy").exists()
    assert (remote / "staging/renamed/quoted ' name.txt").read_text() == "tennis\n"
    recovered = tmp_path / "recovered"
    assert call("download", "staging/renamed", str(recovered), "--verify")["verified"]
    (recovered / "quoted ' name.txt").write_text("corrupt")
    assert call("verify", str(recovered), "staging/renamed", expected=3)["changed"]
    inventory = tmp_path / "inventory.json"
    call("inventory", "--path", "data", "--output", str(inventory))
    records = json.loads(inventory.read_text())["entries"]
    assert any(item["hashes"] for item in records if item["type"] == "file")
    call("inventory", "--path", "data", "--output", str(inventory), expected=1)
    call("trash", "staging/renamed", "--recursive", expected=1)
    assert (remote / "staging/renamed").exists()


def test_overwrite_is_copy_not_deleting_sync(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "input"
    source.mkdir()
    (source / "current").write_text("current")
    destination = remote / "data"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    assert call("upload", str(source), "data", "--overwrite", "--verify")["verified"]
    assert (destination / "keep").read_text() == "keep"
    call("copy", "data", "data/recursive", expected=1)
    call("move", "data", ".", expected=1)
    assert (destination / "current").exists()


def test_single_file_transfer_and_rename(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(256)))
    assert call("upload", str(source), "staging/weights.bin", "--verify")["verified"]
    assert call("copy", "staging/weights.bin", "staging/copy.bin")["verified"]
    assert call("move", "staging/copy.bin", "staging/renamed.bin")["verified"]
    assert (remote / "staging/renamed.bin").read_bytes() == source.read_bytes()
    assert not (remote / "staging/copy.bin").exists()


def test_download_overwrite_preserves_unrelated_local_files(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path], tmp_path: Path
) -> None:
    call, remote = drive_cli
    (remote / "data").mkdir()
    (remote / "data/current").write_text("current")
    destination = tmp_path / "recovered"
    destination.mkdir()
    (destination / "current").write_text("outdated")
    (destination / "keep").write_text("keep")
    call("download", "data", str(destination), "--verify", expected=1)
    assert (destination / "current").read_text() == "outdated"
    assert call("download", "data", str(destination), "--overwrite", "--verify")[
        "verified"
    ]
    assert (destination / "current").read_text() == "current"
    assert (destination / "keep").read_text() == "keep"
    assert call("verify", str(destination), "data", expected=3)["missing_on_drive"] == [
        "keep"
    ]


def test_mkdir_planning_creation_existing_directory_and_file_conflict(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
) -> None:
    call, remote = drive_cli
    assert call("mkdir", "staging/new", "--dry-run")["status"] == "planned"
    assert not (remote / "staging/new").exists()
    assert call("mkdir", "staging/new")["status"] == "completed"
    assert (remote / "staging/new").is_dir()
    (remote / "staging/new/keep").write_text("keep")
    assert call("mkdir", "staging/new")["status"] == "completed"
    assert (remote / "staging/new/keep").read_text() == "keep"
    call("mkdir", "staging/new/keep", expected=1)
    assert (remote / "staging/new/keep").read_text() == "keep"
    call("mkdir", ".", expected=1)


def test_verification_relocation_and_inventory_allow_listed_colons(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path], tmp_path: Path
) -> None:
    call, remote = drive_cli
    source = tmp_path / "input"
    (source / "run:01").mkdir(parents=True)
    name = "run:01/checkpoint:epoch=1.bin"
    (source / name).write_bytes(b"weights")
    assert call("upload", str(source), "data/source", "--verify")["verified"]
    assert call("verify", str(source), "data/source")["matches"]
    assert call("copy", "data/source", "staging/copy")["verified"]
    assert call("move", "staging/copy", "staging/moved")["verified"]
    assert not (remote / "staging/copy").exists()
    assert (remote / "staging/moved" / name).read_bytes() == b"weights"
    inventory = tmp_path / "inventory.json"
    call("inventory", "--path", "staging/moved", "--output", str(inventory))
    entries = json.loads(inventory.read_text())["entries"]
    assert {entry["path"] for entry in entries} == {"run:01", name}


def test_inspect_uses_null_for_unrequested_mime_type_and_limits_hashes(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
) -> None:
    call, remote = drive_cli
    (remote / "weights.bin").write_bytes(b"weights")
    assert call("inspect", "weights.bin")["mime_type"] is None
    metadata = call("inspect", "weights.bin", "--checksum")
    assert metadata["mime_type"]
    assert set(metadata["hashes"]) == {"md5", "sha1", "sha256"}


def test_manifests_and_inventory_only_compute_comparison_hashes(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    drive_module: ModuleType,
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    local = tmp_path / "input"
    local.mkdir()
    (local / "weights.bin").write_bytes(b"weights")
    metadata = call("upload", str(local), "data", "--verify")
    backend = drive_module.RcloneBackend(
        remote_root="test:tennis_lab",
        executable="rclone",
        config_path=Path(metadata["rclone_config"]),
    )
    core = importlib.import_module("drive_core")
    for location in (str(local), str(local / "weights.bin"), "test:tennis_lab/data"):
        _, manifest = core._manifest(backend, location)
        hashes = next(entry[2] for entry in manifest.values() if entry[0] == "file")
        assert set(hashes) == {"md5", "sha1", "sha256"}
    inventory = tmp_path / "inventory.json"
    call("inventory", "--path", "data", "--output", str(inventory))
    document = json.loads(inventory.read_text())
    assert document["rclone_config"] == metadata["rclone_config"]
    assert set(document["entries"][0]["hashes"]) == {"md5", "sha1", "sha256"}
    assert (remote / "data/weights.bin").read_bytes() == b"weights"


def test_verify_download_streams_when_remote_hashes_are_unavailable(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    drive_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    content = bytes(range(256)) * 8192
    local = tmp_path / "weights.bin"
    local.write_bytes(content)
    (remote / "weights.bin").write_bytes(content)
    config_path = call("list")["rclone_config"]
    original_stat = drive_module.RcloneBackend.stat

    def without_remote_hashes(
        self: Any, location: str, *, hashes: bool = False
    ) -> dict[str, Any] | None:
        value: dict[str, Any] | None = original_stat(self, location, hashes=hashes)
        if value is not None and location.startswith("test:"):
            value.pop("Hashes", None)
        return value

    monkeypatch.setattr(drive_module.RcloneBackend, "stat", without_remote_hashes)
    arguments = [
        "--rclone-config",
        config_path,
        "--remote-root",
        "test:tennis_lab",
        "verify",
        str(local),
        "weights.bin",
    ]
    assert drive_module.main(arguments) == 3
    assert json.loads(capsys.readouterr().out)["unverifiable"] == ["."]
    assert drive_module.main([*arguments, "--download"]) == 0
    assert json.loads(capsys.readouterr().out)["matches"]
    local.write_bytes(b"x" * len(content))
    assert drive_module.main([*arguments, "--download"]) == 3
    assert json.loads(capsys.readouterr().out)["changed"] == ["."]


def test_explicit_configuration_overrides_environment(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    drive_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    call, _ = drive_cli
    config_path = call("list")["rclone_config"]
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "missing.conf"))
    assert (
        drive_module.main(
            [
                "--rclone-config",
                config_path,
                "--remote-root",
                "test:tennis_lab",
                "list",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rclone_config"] == config_path


def test_configuration_is_required_even_when_default_files_exist(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    drive_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    call, _ = drive_cli
    config = Path(call("list")["rclone_config"])
    for relative in (
        ".config/rclone/rclone.conf",
        ".config/tennis-lab/rclone-drive.conf",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, target)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)
    assert drive_module.main(["--remote-root", "test:tennis_lab", "list"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["rclone_config"] is None
    assert "--rclone-config" in payload["error"]


def test_missing_explicit_configuration_never_uses_environment_config(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    drive_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    call, _ = drive_cli
    monkeypatch.setenv("RCLONE_CONFIG", call("list")["rclone_config"])
    missing = tmp_path / "missing.conf"
    assert (
        drive_module.main(
            [
                "--rclone-config",
                str(missing),
                "--remote-root",
                "test:tennis_lab",
                "list",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rclone_config"] == str(missing)
    assert "configuration file does not exist" in payload["error"]
