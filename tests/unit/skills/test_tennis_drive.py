"""Failure boundaries for the Drive skill without access to a real remote."""

from __future__ import annotations

import argparse
import importlib
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
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, management = modules
    backend = core.RcloneBackend(remote_root="gdrive:tennis_lab", executable=sys.executable)

    def failed(args: list[str]) -> None:
        raise core.DriveToolError("SECRET_CONFIG_CONTENT")

    monkeypatch.setattr(backend, "json", failed)
    with pytest.raises(core.DriveToolError) as error:
        management._run_trash(argparse.Namespace(path="staging/item"), backend)
    assert "withheld" in str(error.value)
    assert "SECRET" not in str(error.value)
