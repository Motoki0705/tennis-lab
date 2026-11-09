"""Failure boundaries for the Drive skill without access to a real remote."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest

TREE_SOURCE = "outputs/court_detection/evaluate/scene-pair-5500-s384-colab/P12-s42"


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".agents/skills/tennis-drive/scripts"))
    return importlib.import_module("drive_core"), importlib.import_module(
        "drive_management"
    )


@pytest.fixture
def backend(modules: tuple[ModuleType, ModuleType], tmp_path: Path) -> Any:
    core, _ = modules
    config = tmp_path / "rclone.conf"
    config.write_text("[gdrive]\ntype = local\n")
    return core.RcloneBackend(
        remote_root="gdrive:tennis_lab",
        executable=sys.executable,
        config_path=config,
    )


@pytest.fixture
def recording_tree_backend(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path
) -> Any:
    """Run real resolution/snapshot/hash logic; replace just rclone execution."""
    core, _ = modules
    config = tmp_path / "rclone.conf"
    config.touch()

    backend_type: Any = core.RcloneBackend

    class RecordingBackend(backend_type):
        def __init__(self) -> None:
            super().__init__(
                remote_root="gdrive:tennis_lab",
                executable=sys.executable,
                config_path=config,
            )
            self.commands: list[list[str]] = []
            self.fault = ""
            self.readbacks = 0
            self.files = {"evaluation.json": b"report", "domain/sample.png": b"image"}
            self.entries: list[dict[str, Any]] = []
            self.configure(self.files)

        def configure(self, files: dict[str, bytes]) -> None:
            self.files = files
            directories = {
                str(parent)
                for name in files
                for parent in PurePosixPath(name).parents
                if str(parent) != "."
            }
            self.entries = [
                {"Path": name, "IsDir": True, "ID": f"dir-{name}"}
                for name in sorted(directories)
            ] + [
                {
                    "Path": name,
                    "IsDir": False,
                    "ID": f"file-{name}",
                    "Size": len(content),
                    "Hashes": {"MD5": hashlib.md5(content).hexdigest()},
                    "ModTime": "fixed",
                }
                for name, content in files.items()
            ]

        def json(self, args: list[str]) -> Any:
            self.commands.append(args)
            assert args[0] == "lsjson", args
            if "--recursive" in args:
                self.readbacks += 1
                entries = copy.deepcopy(self.entries)
                if self.fault == "duplicate-directory":
                    entries.append(
                        {"Path": "domain", "IsDir": True, "ID": "duplicate-dir"}
                    )
                elif self.fault == "duplicate-file":
                    entries.append(copy.deepcopy(entries[-1]))
                elif self.fault == "missing":
                    entries = [
                        entry for entry in entries if entry["Path"] != "evaluation.json"
                    ]
                elif self.fault == "extra":
                    entries.append(
                        {
                            "Path": "extra",
                            "IsDir": False,
                            "Size": 0,
                            "Hashes": {"MD5": hashlib.md5(b"").hexdigest()},
                        }
                    )
                elif self.fault == "unverifiable":
                    entries[-1]["Hashes"] = {}
                elif self.fault == "modified":
                    entries[-1]["Hashes"]["MD5"] = hashlib.md5(b"other").hexdigest()
                elif self.fault == "during-transfer" and self.readbacks > 1:
                    entries[-1]["ID"] = "replacement-with-same-content"
                elif self.fault == "unsafe":
                    entries[-1]["Path"] = "../escaped"
                return entries
            parts = ["tennis_lab", *PurePosixPath(TREE_SOURCE).parts]
            parent = "gdrive:"
            for index, name in enumerate(parts):
                if args[1] == parent:
                    entry = {"Name": name, "IsDir": True, "ID": name}
                    duplicate = self.fault == "duplicate-ancestor" and index == 0
                    duplicate |= (
                        self.fault == "late-duplicate-ancestor"
                        and self.readbacks > 0
                        and index == 0
                    )
                    return (
                        [entry, {**entry, "ID": "duplicate-root"}]
                        if duplicate
                        else [entry]
                    )
                parent = (
                    parent.rstrip("/") + ("" if parent.endswith(":") else "/") + name
                )
            prefix = self.remote_path(TREE_SOURCE)
            if args[1] == prefix or args[1].startswith(prefix + "/"):
                parent_name = args[1][len(prefix) :].lstrip("/") or "."
                return [
                    {**entry, "Name": PurePosixPath(entry["Path"]).name}
                    for entry in self.entries
                    if str(PurePosixPath(entry["Path"]).parent) == parent_name
                ]
            raise AssertionError(f"Unexpected listing: {args}")

        def stat(self, location: str, *, hashes: bool = False) -> dict[str, Any]:
            self.commands.append(["lsjson", location, "--stat"])
            prefix = self.remote_path(TREE_SOURCE) + "/"
            content = (
                self.files[location[len(prefix) :]]
                if location.startswith(prefix)
                else Path(location).read_bytes()
            )
            return {
                "IsDir": False,
                "Size": len(content),
                "Hashes": {"MD5": hashlib.md5(content).hexdigest()},
            }

        def run(self, args: list[str], **kwargs: Any) -> None:
            self.commands.append(args)
            assert args[0] in {"copy", "copyto"}, args
            target = Path(args[2])
            if args[0] == "copyto":
                name = args[1][len(self.remote_path(TREE_SOURCE)) + 1 :]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.files[name])
                return
            for name, content in self.files.items():
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            if self.fault == "corrupt-transfer":
                (target / "domain/sample.png").write_bytes(b"other")
            kwargs["progress"](
                {
                    "transfers": len(self.files),
                    "bytes": sum(map(len, self.files.values())),
                }
            )

    return RecordingBackend()


@pytest.mark.parametrize("file_count", [1, 259])
def test_tree_download_batches_rclone_calls_and_reports_verified_bytes(
    recording_tree_backend: Any, tmp_path: Path, file_count: int
) -> None:
    tree = importlib.import_module("drive_tree")
    backend = recording_tree_backend
    files = {
        f"domain/{index:03d}.png": bytes([index % 256]) * (index + 1)
        for index in range(file_count)
    }
    backend.configure(files)
    progress: list[dict[str, Any]] = []
    expected = {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    destination = tmp_path / "recovered"
    receipt = tree.download_tree(
        backend,
        TREE_SOURCE,
        destination,
        expected_files=expected,
        progress=lambda **fields: progress.append(fields),
    )
    # Six path components including tennis_lab: 12 ancestor listings, two
    # recursive snapshots, one copy. No local rclone stat or per-file process.
    assert len(backend.commands) == 15
    assert sum(args[0] == "copy" for args in backend.commands) == 1
    assert sum("--recursive" in args for args in backend.commands) == 2
    assert not any("--stat" in args for args in backend.commands)
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*.png")
    } == files
    assert receipt["verified"] is True
    assert receipt["total_files"] == file_count
    assert receipt["total_bytes"] == sum(map(len, files.values()))
    assert {event["completed_files"] for event in progress} == set(
        range(file_count + 1)
    )
    assert progress[-1]["completed_bytes"] == sum(map(len, files.values()))
    assert progress[-1]["operation"] == "completed"


@pytest.mark.parametrize("file_count", [1, 259])
def test_single_file_download_retains_its_existing_verification_contract(
    recording_tree_backend: Any,
    tmp_path: Path,
    file_count: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    drive = importlib.import_module("drive")
    backend = recording_tree_backend
    backend.configure(
        {f"domain/{index:03d}.png": b"image" for index in range(file_count)}
    )
    for name in backend.files:
        args = drive.build_parser().parse_args(
            [
                "download",
                f"{TREE_SOURCE}/{name}",
                str(tmp_path / "old" / name),
                "--verify",
            ]
        )
        assert drive._run_download(args, backend) == 0
    capsys.readouterr()
    # Keep the existing eight-component file path preflight/readback and hashes.
    assert len(backend.commands) == 19 * file_count
    assert sum(args[0] == "copyto" for args in backend.commands) == file_count
    assert sum("--stat" in args for args in backend.commands) == 2 * file_count


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("duplicate-ancestor", "Ambiguous"),
        ("duplicate-directory", "Ambiguous"),
        ("duplicate-file", "Ambiguous"),
        ("late-duplicate-ancestor", "Ambiguous"),
        ("missing", "file set"),
        ("extra", "file set"),
        ("unverifiable", "hashes"),
        ("modified", "hash changed"),
        ("corrupt-transfer", "hash changed"),
        ("during-transfer", "changed during"),
        ("unsafe", "Unsafe"),
    ],
)
def test_tree_download_refuses_ambiguous_changed_and_corrupt_content(
    recording_tree_backend: Any, tmp_path: Path, fault: str, message: str
) -> None:
    tree = importlib.import_module("drive_tree")
    backend = recording_tree_backend
    backend.fault = fault
    expected = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in backend.files.items()
    }
    destination = tmp_path / "recovered"
    with pytest.raises(tree.DriveToolError, match=message):
        tree.download_tree(backend, TREE_SOURCE, destination, expected_files=expected)
    assert not destination.exists()
    assert not list(tmp_path.glob(".recovered.fetching-*"))


def test_tree_download_checks_pinned_sha_even_when_drive_hash_matches(
    recording_tree_backend: Any, tmp_path: Path
) -> None:
    tree = importlib.import_module("drive_tree")
    backend = recording_tree_backend
    expected = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in backend.files.items()
    }
    backend.configure({**backend.files, "domain/sample.png": b"other"})
    with pytest.raises(tree.DriveToolError, match="Expected subtree SHA-256 changed"):
        tree.download_tree(
            backend, TREE_SOURCE, tmp_path / "recovered", expected_files=expected
        )


def test_tree_existing_directory_is_verified_without_overwriting(
    recording_tree_backend: Any, tmp_path: Path
) -> None:
    tree = importlib.import_module("drive_tree")
    backend = recording_tree_backend
    destination = tmp_path / "recovered"
    tree.download_tree(backend, TREE_SOURCE, destination)
    backend.commands.clear()
    assert tree.download_tree(backend, TREE_SOURCE, destination)["reused"] is True
    assert len(backend.commands) == 14
    (destination / "domain/sample.png").write_bytes(b"other")
    with pytest.raises(tree.DriveToolError, match="hash changed"):
        tree.download_tree(backend, TREE_SOURCE, destination)
    assert (destination / "domain/sample.png").read_bytes() == b"other"
    assert not any(args[0] == "copy" for args in backend.commands)


def test_tree_cli_publishes_local_progress_and_refuses_stale_id(
    recording_tree_backend: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    drive = importlib.import_module("drive")
    backend = recording_tree_backend
    monkeypatch.setattr(drive, "RcloneBackend", lambda **kwargs: backend)
    progress = tmp_path / "progress.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in backend.files.items()
                },
            }
        )
    )
    args = [
        "--rclone-config",
        str(backend.config_path),
        "download-tree",
        TREE_SOURCE,
        str(tmp_path / "recovered"),
        "--expected-manifest",
        str(manifest),
        "--progress-file",
        str(progress),
    ]
    assert (
        drive.main(
            [*args, "--progress-file", str(tmp_path / "recovered/progress.json")]
        )
        == 1
    )
    assert "outside" in json.loads(capsys.readouterr().out)["error"]
    assert not (tmp_path / "recovered").exists()
    assert drive.main([*args, "--expected-id", "stale"]) == 1
    assert "ID changed" in json.loads(capsys.readouterr().out)["error"]
    assert json.loads(progress.read_text())["operation"] == "failed"
    assert drive.main(args) == 0
    assert json.loads(capsys.readouterr().out)["verified"]
    value = json.loads(progress.read_text())
    assert value["operation"] == "completed"
    assert value["completed_files"] == value["total_files"] == 2
    assert value["completed_bytes"] == value["total_bytes"] == 11
    assert value["last_progress_at"] <= value["updated_at"]


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
    backend: Any,
) -> None:
    core, _ = modules
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
    backend: Any,
) -> None:
    core, _ = modules
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
    backend: Any,
) -> None:
    core, management = modules
    with pytest.raises(core.DriveToolError, match="project root"):
        management._child(backend, path)


def test_trash_requires_drive_and_forces_trash_even_if_config_disables_it(
    modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    backend: Any,
) -> None:
    core, management = modules
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
    backend: Any,
) -> None:
    core, management = modules
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
    backend: Any,
) -> None:
    core, management = modules

    def failed(args: list[str]) -> None:
        raise core.DriveToolError("SECRET_CONFIG_CONTENT")

    monkeypatch.setattr(backend, "json", failed)
    with pytest.raises(core.DriveToolError) as error:
        management._run_trash(argparse.Namespace(path="staging/item"), backend)
    assert "withheld" in str(error.value)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("path", ["../data", "/data", "data/../../x"])
def test_listed_paths_cannot_escape_the_subtree(
    modules: tuple[ModuleType, ModuleType], path: str
) -> None:
    core, _ = modules
    with pytest.raises(core.DriveToolError):
        core.RcloneBackend.normalize_listed_relative(path)


@pytest.fixture
def streaming_backend(backend: Any, tmp_path: Path) -> Any:
    executable = tmp_path / "rclone-stream"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "if sys.argv[1] != '--config' or not pathlib.Path(sys.argv[2]).is_file():\n"
        "    sys.exit(10)\n"
        "if sys.argv[3] != 'cat':\n"
        "    sys.exit(11)\n"
        "if sys.argv[4].endswith('/timeout'):\n"
        "    time.sleep(30)\n"
        "if sys.argv[4].endswith('/failure'):\n"
        "    sys.stderr.write('fixture stream failed')\n"
        "    sys.exit(9)\n"
        "sys.stdout.buffer.write(b'tennis' * 200000)\n"
    )
    executable.chmod(0o700)
    backend.executable = str(executable)
    backend.timeout = 5
    return backend


def test_streaming_remote_checksum_reads_all_content(
    modules: tuple[ModuleType, ModuleType], streaming_backend: Any
) -> None:
    core, _ = modules
    assert (
        core._sha256_remote(streaming_backend, "gdrive:tennis_lab/content")
        == hashlib.sha256(b"tennis" * 200000).hexdigest()
    )


def test_streaming_remote_checksum_rejects_nonzero_exit(
    modules: tuple[ModuleType, ModuleType], streaming_backend: Any
) -> None:
    core, _ = modules
    with pytest.raises(core.DriveToolError, match="fixture stream failed"):
        core._sha256_remote(streaming_backend, "gdrive:tennis_lab/failure")


def test_streaming_remote_checksum_kills_on_timeout(
    modules: tuple[ModuleType, ModuleType], streaming_backend: Any
) -> None:
    core, _ = modules
    streaming_backend.timeout = 1
    started = time.monotonic()
    with pytest.raises(core.DriveToolError, match="timed out"):
        core._sha256_remote(streaming_backend, "gdrive:tennis_lab/timeout")
    assert streaming_backend.timeout <= time.monotonic() - started < 10


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_transfer_stats_use_the_same_process_and_keep_failure_boundaries(
    modules: tuple[ModuleType, ModuleType],
    backend: Any,
    tmp_path: Path,
    outcome: str,
) -> None:
    core, _ = modules
    executable = tmp_path / "rclone-stats"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "assert '--use-json-log' in sys.argv\n"
        "sys.stderr.write(json.dumps({'stats': {'transfers': 1, 'bytes': 7}}) + '\\n')\n"
        "sys.stderr.flush()\n"
        "if sys.argv[4] == 'timeout': time.sleep(30)\n"
        "if sys.argv[4] == 'failure':\n"
        "    sys.stderr.write('injected transfer failure\\n')\n"
        "    sys.exit(9)\n"
    )
    executable.chmod(0o700)
    backend.executable = str(executable)
    backend.timeout = 1
    observations: list[dict[str, Any]] = []
    if outcome == "success":
        result = backend.run(["copy", outcome], progress=observations.append)
        assert result.returncode == 0
    else:
        with pytest.raises(core.DriveToolError, match="transfer failure|timed out"):
            backend.run(["copy", outcome], progress=observations.append)
    assert observations == [{"transfers": 1, "bytes": 7}]
