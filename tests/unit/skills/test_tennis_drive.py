"""Failure boundaries for the Drive skill without access to a real remote."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".agents/skills/tennis-drive/scripts"))
    return importlib.import_module("drive_core"), importlib.import_module(
        "drive_management"
    )


@pytest.mark.parametrize(
    "path", ["../data", "/data", "data/../../x", "x:y", "x\\y", "x\ny"]
)
def test_root_escape_rejected(
    modules: tuple[ModuleType, ModuleType], path: str
) -> None:
    core, _ = modules
    with pytest.raises(core.DriveToolError):
        core.RcloneBackend.normalize_relative(path)


def test_duplicate_ancestor_stops_before_descending(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _ = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    calls: list[list[str]] = []

    def listing(args: list[str]) -> list[dict[str, Any]]:
        calls.append(args)
        return [
            {"Name": "tennis_lab", "IsDir": True, "ID": "first"},
            {"Name": "tennis_lab", "IsDir": True, "ID": "second"},
        ]

    monkeypatch.setattr(backend, "json", listing)
    with pytest.raises(core.DriveToolError, match="Ambiguous.*first.*second"):
        backend.resolve("data/new", must_exist=False)
    assert len(calls) == 1


def test_duplicate_leaf_is_not_silently_collapsed_in_verification(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _ = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    monkeypatch.setattr(backend, "stat", lambda *a, **k: {"IsDir": True})
    monkeypatch.setattr(
        backend,
        "json",
        lambda args: [
            {"Path": "a.bin", "IsDir": False, "Size": 1, "Hashes": {"MD5": "aaa"}},
            {"Path": "a.bin", "IsDir": False, "Size": 1, "Hashes": {"MD5": "bbb"}},
        ],
    )
    with pytest.raises(core.DriveToolError, match="Duplicate path"):
        core._manifest(backend, "gdrive:tennis_lab/data")


@pytest.mark.parametrize("path", [".", "", "./"])
def test_management_cannot_target_project_root(
    modules: tuple[ModuleType, ModuleType],
    path: str,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    with pytest.raises(core.DriveToolError, match="project root"):
        management._child(backend, path)


def test_trash_requires_drive_and_forces_trash_even_if_config_disables_it(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    arguments = argparse.Namespace(
        path="staging/file",
        recursive=False,
        expected_id="item-1",
        dry_run=False,
        format="json",
    )
    monkeypatch.setattr(
        backend,
        "json",
        lambda args: {"gdrive": {"type": "drive", "use_trash": "false"}},
    )
    monkeypatch.setattr(
        backend, "resolve", lambda path: {"ID": "item-1", "IsDir": False}
    )
    monkeypatch.setattr(backend, "stat", lambda path: None)
    mutations: list[list[str]] = []
    monkeypatch.setattr(backend, "run", lambda args, **kwargs: mutations.append(args))
    assert management._run_trash(arguments, backend) == 0
    assert mutations == [
        ["deletefile", "gdrive:tennis_lab/staging/file", "--drive-use-trash=true"]
    ]
    monkeypatch.setattr(backend, "json", lambda args: {"gdrive": {"type": "local"}})
    with pytest.raises(core.DriveToolError, match="direct Google Drive"):
        management._run_trash(arguments, backend)
    assert len(mutations) == 1


def test_stale_identity_and_directory_without_recursive_never_mutate(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    args = argparse.Namespace(
        path="staging/data",
        expected_id="old",
        recursive=False,
        dry_run=False,
        format="json",
    )
    monkeypatch.setattr(backend, "json", lambda args: {"gdrive": {"type": "drive"}})
    monkeypatch.setattr(backend, "resolve", lambda path: {"ID": "new", "IsDir": True})
    mutations: list[object] = []
    monkeypatch.setattr(backend, "run", lambda *a, **k: mutations.append(a))
    with pytest.raises(core.DriveToolError, match="ID changed"):
        management._run_trash(args, backend)
    args.expected_id = "new"
    with pytest.raises(core.DriveToolError, match="recursive"):
        management._run_trash(args, backend)
    args.recursive = True
    args.dry_run = True
    assert management._run_trash(args, backend) == 0
    assert not mutations


def test_size_without_hash_does_not_prove_content(
    modules: tuple[ModuleType, ModuleType],
) -> None:
    core, management = modules
    before = {"a": ("file", 12, {"md5": "abcd"})}
    assert not management._same_content(before, {"a": ("file", 12, {})})
    assert not management._same_content(before, {"a": ("file", 12, {"md5": "efgh"})})
    with pytest.raises(core.DriveToolError, match="hashes"):
        management._require_hashes({"a": ("file", 12, {})})


def test_failed_config_read_never_exposes_configuration(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )

    def failed(args: list[str]) -> None:
        raise core.DriveToolError("SECRET_CONFIG_CONTENT")

    monkeypatch.setattr(backend, "json", failed)
    with pytest.raises(core.DriveToolError) as error:
        management._run_trash(argparse.Namespace(path="staging/item"), backend)
    assert "withheld" in str(error.value)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "configuration", [[], {"gdrive": None}, {"gdrive": "SECRET_CONFIG_CONTENT"}]
)
def test_malformed_config_is_refused_without_exposing_it_or_mutating(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    configuration: Any,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    monkeypatch.setattr(backend, "json", lambda args: configuration)
    mutations: list[object] = []
    monkeypatch.setattr(backend, "run", lambda *a, **k: mutations.append(a))
    with pytest.raises(core.DriveToolError, match="direct Google Drive") as error:
        management._run_trash(argparse.Namespace(path="staging/item"), backend)
    assert "SECRET" not in str(error.value)
    assert not mutations


@pytest.mark.parametrize("path", ["Copy of x: v2", "a/back\\slash", "a/ISO 12:34:56"])
def test_manifest_preserves_backend_file_names(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    core, _ = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    monkeypatch.setattr(backend, "stat", lambda *a, **k: {"IsDir": True})
    monkeypatch.setattr(
        backend,
        "json",
        lambda args: [
            {"Path": path, "IsDir": False, "Size": 1, "Hashes": {"MD5": "aaa"}}
        ],
    )
    kind, entries = core._manifest(backend, "gdrive:tennis_lab/data")
    assert kind == "directory"
    assert entries == {path: ("file", 1, {"md5": "aaa"})}


@pytest.mark.parametrize("path", ["", "/x", "../x", "a/../x", "a//x", "a/", "./x"])
def test_manifest_rejects_unsafe_backend_paths(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    core, _ = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    monkeypatch.setattr(backend, "stat", lambda *a, **k: {"IsDir": True})
    monkeypatch.setattr(
        backend, "json", lambda args: [{"Path": path, "IsDir": False, "Size": 1}]
    )
    with pytest.raises(core.DriveToolError, match="path"):
        core._manifest(backend, "gdrive:tennis_lab/data")


def test_inventory_records_root_and_child_identities(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core, management = modules
    backend = core.RcloneBackend(
        remote_root="gdrive:tennis_lab", executable=sys.executable
    )
    monkeypatch.setattr(
        backend, "resolve", lambda path: {"ID": "root-id", "IsDir": True}
    )
    monkeypatch.setattr(
        backend,
        "json",
        lambda args: [
            {"Path": "child: v2", "ID": "child-id", "IsDir": False, "Size": 1}
        ],
    )
    output = tmp_path / "inventory.json"
    args = argparse.Namespace(path="data", output=str(output), format="json")
    assert management._run_inventory(args, backend) == 0
    document = json.loads(output.read_text())
    assert document["path"] == "data"
    assert document["root_id"] == "root-id"
    assert document["entries"][0]["id"] == "child-id"
    assert document["entries"][0]["path"] == "child: v2"
    assert json.loads(capsys.readouterr().out)["root_id"] == "root-id"


def test_transfer_failure_returns_all_verification_details(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core, _ = modules
    cli = importlib.import_module("drive")
    source = tmp_path / "source"
    source.mkdir()
    resolved = iter([None, {"IsDir": True}])
    monkeypatch.setattr(core.RcloneBackend, "resolve", lambda *a, **k: next(resolved))
    monkeypatch.setattr(core.RcloneBackend, "run", lambda *a, **k: None)
    verification = core.Verification(
        matches=False,
        local_type="directory",
        drive_type="directory",
        changed=["changed"],
        missing_on_drive=["missing"],
        extra_on_drive=["extra"],
        unverifiable=["unverifiable"],
    )
    monkeypatch.setattr(cli, "_verify_paths", lambda **kwargs: verification)
    assert (
        cli.main(
            ["--rclone-bin", sys.executable, "upload", str(source), "data", "--verify"]
        )
        == 1
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert not payload["ok"]
    assert payload["verification"] == {
        "matches": False,
        "local_type": "directory",
        "drive_type": "directory",
        "changed": ["changed"],
        "missing_on_drive": ["missing"],
        "extra_on_drive": ["extra"],
        "unverifiable": ["unverifiable"],
    }
    assert not captured.err


@pytest.mark.parametrize(
    ("command", "response"),
    [
        (["list"], {}),
        (["list"], [None]),
        (["inventory"], [{"IsDir": False, "Size": 1}]),
        (["list"], [{"Path": 12, "IsDir": False, "Size": 1}]),
        (["list"], [{"Path": "a", "IsDir": "false", "Size": 1}]),
        (["list"], [{"Path": "a", "IsDir": False, "Size": "invalid"}]),
        (["list"], [{"Path": "a", "IsDir": False, "Size": None}]),
        (["list"], [{"Path": "a", "IsDir": False, "Size": 1, "Hashes": []}]),
        (
            ["inspect", ".", "--checksum"],
            {"Path": "tennis_lab", "IsDir": False, "Size": "invalid"},
        ),
        (["inspect", "."], {"count": "invalid", "bytes": 1}),
        (["inspect", "."], []),
    ],
)
def test_malformed_rclone_responses_return_json_errors(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    response: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core, _ = modules
    cli = importlib.import_module("drive")

    def run(
        backend: Any, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        value = response
        if arguments[:2] == ["lsjson", "gdrive:"]:
            value = [
                {"Name": "tennis_lab", "Path": "tennis_lab", "IsDir": True, "Size": -1}
            ]
        return subprocess.CompletedProcess(arguments, 0, stdout=json.dumps(value))

    monkeypatch.setattr(core.RcloneBackend, "run", run)
    if command == ["inventory"]:
        command = [*command, "--output", str(tmp_path / "inventory.json")]
    assert (
        cli.main(
            [
                "--remote-root",
                "gdrive:tennis_lab",
                "--rclone-bin",
                sys.executable,
                *command,
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert not payload["ok"]
    assert "rclone" in payload["error"]
    assert not captured.err
    assert not (tmp_path / "inventory.json").exists()
