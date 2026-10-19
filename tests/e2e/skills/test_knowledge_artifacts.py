"""CPU fixtures for run retention, immutable exceptions and deletion provenance."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/knowledge-control/scripts"


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("kg_artifacts")


def write(path: Path, value: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def save(path: Path, data: dict[str, Any]) -> Path:
    return write(path, (json.dumps(data) + "\n").encode())


def deletion(
    base: Path, commit: str = "a" * 40, content: bytes = b"prediction"
) -> Path:
    return save(
        base / "runs/run-fixture/artifact-manifest.json",
        {
            "schema_version": 1,
            "run_id": "run-fixture",
            "files": [
                {
                    "path": "predictions/pred.npz",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "reason": "raw-data-events-archives",
                    "removed_from_commit": commit,
                }
            ],
        },
    )


def exceptions(
    base: Path,
    *,
    files: list[dict[str, str]] | None = None,
    runs: list[dict[str, str]] | None = None,
    legacy: list[dict[str, str]] | None = None,
) -> Path:
    return save(
        base / "run-storage-exceptions.json",
        {
            "schema_version": 1,
            "files": files or [],
            "runs": runs or [],
            "legacy_manifests": legacy or [],
        },
    )


@pytest.mark.parametrize(
    "suffix,limit",
    [
        (".json", 256 * 1024),
        (".patch", 256 * 1024),
        (".png", 512 * 1024),
        (".svg", 512 * 1024),
        (".pdf", 512 * 1024),
    ],
)
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_file_size_boundaries(
    tmp_path: Path, artifacts: Any, suffix: str, limit: int, delta: int
) -> None:
    write(tmp_path / f"runs/run-fixture/artifact{suffix}", b"x" * (limit + delta))
    errors = artifacts.validate_runs(tmp_path, [])
    assert bool(errors) == (delta > 0)
    if errors:
        assert any("over-" in error for error in errors)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_run_total_boundary(tmp_path: Path, artifacts: Any, delta: int) -> None:
    for index in range(8):
        write(
            tmp_path / f"runs/run-fixture/{index}.txt",
            b"x" * (256 * 1024 - (1 if index == 0 and delta == -1 else 0)),
        )
    if delta == 1:
        write(tmp_path / "runs/run-fixture/extra.txt", b"x")
    errors = artifacts.validate_runs(tmp_path, [])
    assert bool(errors) == (delta > 0)
    if errors:
        assert any("run exceeds 2 MiB" in error for error in errors)


@pytest.mark.parametrize(
    "name",
    [
        "pred.npz",
        "pred.npy",
        "weights.ckpt",
        "weights.pt",
        "weights.safetensors",
        "events.out.tfevents.fixture",
        "log.gz",
        "log.zip",
        "preview.mp4",
        "preview.gif",
        "contact.jpg",
        "contact.jpeg",
        "timing.prof",
        "unknown.bin",
    ],
)
def test_raw_or_unknown_types_are_rejected_even_when_small(
    tmp_path: Path, artifacts: Any, name: str
) -> None:
    write(tmp_path / "runs/run-fixture" / name)
    assert any("run storage policy" in e for e in artifacts.validate_runs(tmp_path, []))


def test_symlink_is_rejected(tmp_path: Path, artifacts: Any) -> None:
    target = write(tmp_path / "target.txt")
    directory = tmp_path / "runs/run-fixture"
    directory.mkdir(parents=True)
    (directory / "alias.txt").symlink_to(target)
    assert any(
        "symlinks are not allowed" in e for e in artifacts.validate_runs(tmp_path, [])
    )


def test_exceptions_pin_file_content_and_entire_run(
    tmp_path: Path, artifacts: Any
) -> None:
    directory = tmp_path / "runs/run-fixture"
    patch = write(directory / "uncommitted.patch", b"x" * (2 * 1024 * 1024 + 1))
    file_row = {
        "path": "runs/run-fixture/uncommitted.patch",
        "sha256": artifacts.sha256(patch),
        "reason": "Existing reproduction patch",
    }
    run_row = {
        "path": "runs/run-fixture",
        "sha256": artifacts.tree_sha256(directory),
        "reason": "Existing bundle, frozen files",
    }
    exceptions(tmp_path, files=[file_row], runs=[run_row])
    assert artifacts.validate_runs(tmp_path, []) == []
    write(directory / "extra.txt", b"x")
    assert any(
        "run exception SHA-256 mismatch" in e
        for e in artifacts.validate_runs(tmp_path, [])
    )
    (directory / "extra.txt").unlink()
    patch.write_bytes(b"y" * patch.stat().st_size)
    assert any(
        "exception SHA-256 mismatch" in e for e in artifacts.validate_runs(tmp_path, [])
    )
    patch.rename(directory / "renamed.patch")
    errors = artifacts.validate_runs(tmp_path, [])
    assert any("stale files exception" in e for e in errors)
    assert any("file-over-256KiB" in e for e in errors)


@pytest.mark.parametrize(
    "change,expected",
    [
        ("version", "unsupported schema_version"),
        ("duplicate", "duplicate exception path"),
        ("reason", "reason is required"),
        ("hash", "invalid exception SHA-256"),
        ("escape", "invalid or duplicate exception path"),
    ],
)
def test_exception_list_validation(
    tmp_path: Path, artifacts: Any, change: str, expected: str
) -> None:
    row = {
        "path": "runs/run-fixture/old.patch",
        "sha256": "a" * 64,
        "reason": "Existing",
    }
    path = exceptions(tmp_path, files=[row])
    data = json.loads(path.read_text())
    if change == "version":
        data["schema_version"] = 2
    elif change == "duplicate":
        data["files"].append(row)
    elif change == "reason":
        data["files"][0]["reason"] = " "
    elif change == "hash":
        data["files"][0]["sha256"] = "bad"
    else:
        data["files"][0]["path"] = "runs/../outside.patch"
    save(path, data)
    assert any(expected in error for error in artifacts.validate_runs(tmp_path, []))


def test_legacy_inventory_requires_explicit_path_and_hash(
    tmp_path: Path, artifacts: Any
) -> None:
    path = save(
        tmp_path / "runs/run-fixture/artifact-manifest.json",
        {"schema_version": 1, "kind": "historical_inventory", "files": []},
    )
    assert artifacts.validate_runs(tmp_path, [])
    row = {
        "path": path.relative_to(tmp_path).as_posix(),
        "sha256": artifacts.sha256(path),
        "reason": "Keep the historical inventory verbatim",
    }
    exceptions(tmp_path, legacy=[row])
    assert artifacts.validate_runs(tmp_path, []) == []
    path.write_text(path.read_text() + " ")
    assert any(
        "legacy manifest SHA-256 mismatch" in e
        for e in artifacts.validate_runs(tmp_path, [])
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ("version", "unsupported schema_version"),
        ("path", "invalid or duplicate removed path"),
        ("hash", "invalid SHA-256"),
        ("commit", "full pre-deletion HEAD"),
        ("missing", "invalid deletion record fields"),
        ("destination", "invalid manifest fields"),
        ("reason", "unknown deletion reason"),
        ("type_reason", "disagrees with type/size"),
        ("present", "deleted file still exists"),
    ],
)
def test_manifest_rejects_invalid_or_inconsistent_records(
    tmp_path: Path, artifacts: Any, change: str, expected: str
) -> None:
    path = deletion(tmp_path)
    data = json.loads(path.read_text())
    record = data["files"][0]
    if change == "version":
        data["schema_version"] = 99
    elif change == "path":
        record["path"] = "../outside.npz"
    elif change == "hash":
        record["sha256"] = "wrong"
    elif change == "commit":
        record["removed_from_commit"] = "HEAD"
    elif change == "missing":
        record.pop("sha256")
    elif change == "destination":
        data["destination"] = "not permitted"
    elif change == "reason":
        record["reason"] = "forgot"
    elif change == "type_reason":
        record["reason"] = "figure-over-512KiB"
    else:
        write(path.parent / record["path"])
    save(path, data)
    assert any(expected in error for error in artifacts.validate_runs(tmp_path, []))


def node(artifacts: Any, base: Path, metadata: dict[str, Any], body: str = "") -> Any:
    lib = importlib.import_module("kg_lib")
    return lib.Node(
        "run-fixture",
        "run",
        {"artifacts": metadata},
        body,
        base / "nodes/topic/000001-run-fixture.md",
    )


def test_deleted_file_directory_and_body_references(
    tmp_path: Path, artifacts: Any
) -> None:
    path = deletion(tmp_path)
    record = "knowledge/runs/run-fixture/artifact-manifest.json#/files/0"
    meta = {
        "predictions": {"status": "deleted", "record": record},
        "prediction_directory": {
            "status": "deleted",
            "record": record.split("#")[0],
            "path_prefix": "predictions/",
        },
        "output_dir": "outputs/unavailable",
        "checkpoint": "/not-in-ci/weights.ckpt",
    }
    body = (
        "[生予測（削除済み）](../../runs/run-fixture/artifact-manifest.json#/files/0)"
    )
    fixture = node(artifacts, tmp_path, meta, body)
    assert artifacts.validate_runs(tmp_path, [fixture]) == []
    fixture.meta["artifacts"]["predictions"] = (
        "knowledge/runs/run-fixture/predictions/pred.npz"
    )
    assert any(
        "broken run reference" in e
        for e in artifacts.validate_runs(tmp_path, [fixture])
    )
    fixture.meta["artifacts"]["predictions"] = {
        "status": "deleted",
        "record": record.replace("/0", "/5"),
    }
    assert any(
        "missing deletion record" in e
        for e in artifacts.validate_runs(tmp_path, [fixture])
    )
    fixture.meta["artifacts"]["predictions"] = {"status": "deleted", "record": record}
    fixture.body = body.replace("（削除済み）", "")
    assert any(
        "must be labelled" in e for e in artifacts.validate_runs(tmp_path, [fixture])
    )
    fixture.body = body
    path.unlink()
    assert any(
        "missing deletion record" in e
        for e in artifacts.validate_runs(tmp_path, [fixture])
    )


def test_relative_bare_and_nested_broken_references(
    tmp_path: Path, artifacts: Any
) -> None:
    write(tmp_path / "runs/run-fixture/metrics.json", b"{}")
    fixture = node(
        artifacts,
        tmp_path,
        {"nested": ["knowledge/runs/run-fixture/metrics.json"]},
        "[metrics](../../runs/run-fixture/metrics.json) `knowledge/runs/run-fixture/metrics.json` `knowledge/runs/*`",
    )
    assert artifacts.validate_runs(tmp_path, [fixture]) == []
    fixture.body = fixture.body.replace("metrics.json", "missing.json")
    assert len(artifacts.validate_runs(tmp_path, [fixture])) == 2


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_history_checks_missing_record_byte_identity_and_immutability(
    tmp_path: Path, artifacts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "knowledge"
    raw = write(base / "runs/run-fixture/predictions/pred.npz", b"actual prediction")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "base",
    )
    commit = git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(artifacts, "repo_root", lambda: tmp_path)
    raw.unlink()
    errors = artifacts.validate_runs(base, [], "HEAD")
    assert any("missing its deletion record" in error for error in errors)
    path = deletion(base, commit, b"actual prediction")
    assert artifacts.validate_runs(base, [], "HEAD") == []
    data = json.loads(path.read_text())
    data["files"][0]["sha256"] = "0" * 64
    save(path, data)
    assert any(
        "SHA-256 mismatch with base blob" in e
        for e in artifacts.validate_runs(base, [], "HEAD")
    )
    data["files"][0]["sha256"] = hashlib.sha256(b"actual prediction").hexdigest()
    data["files"][0]["size_bytes"] += 1
    save(path, data)
    assert any(
        "size/SHA-256 mismatch" in e for e in artifacts.validate_runs(base, [], "HEAD")
    )
    deletion(base, commit, b"actual prediction")
    git(tmp_path, "add", "-A")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "record deletion",
    )
    assert artifacts.validate_runs(base, [], "HEAD") == []
    data = json.loads(path.read_text())
    data["files"][0]["sha256"] = "0" * 64
    save(path, data)
    assert any(
        "records are immutable" in e for e in artifacts.validate_runs(base, [], "HEAD")
    )
    path.unlink()
    assert any(
        "missing its deletion record" in e
        for e in artifacts.validate_runs(base, [], "HEAD")
    )


def test_recorded_commit_must_contain_the_file(
    tmp_path: Path, artifacts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "knowledge"
    write(tmp_path / "README.md")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "earlier",
    )
    earlier = git(tmp_path, "rev-parse", "HEAD")
    raw = write(base / "runs/run-fixture/predictions/pred.npz", b"prediction")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "base",
    )
    monkeypatch.setattr(artifacts, "repo_root", lambda: tmp_path)
    raw.unlink()
    deletion(base, earlier)
    assert any(
        "does not contain the base blob" in e
        for e in artifacts.validate_runs(base, [], "HEAD")
    )


def test_existing_records_do_not_require_erased_history(
    tmp_path: Path, artifacts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "knowledge"
    deletion(base)  # The old source commit does not exist in this repository.
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-qm",
        "rewritten snapshot",
    )
    monkeypatch.setattr(artifacts, "repo_root", lambda: tmp_path)
    assert artifacts.validate_runs(base, [], "HEAD") == []


@pytest.mark.parametrize(
    "text",
    [
        "[evidence]: ../../runs/run-fixture/missing.json",
        "`../../runs/run-fixture/missing.json`",
    ],
)
def test_other_markdown_run_references_are_checked(
    tmp_path: Path, artifacts: Any, text: str
) -> None:
    fixture = node(artifacts, tmp_path, {}, text)
    assert any(
        "broken run reference" in e
        for e in artifacts.validate_runs(tmp_path, [fixture])
    )


@pytest.mark.parametrize(
    "name,size",
    [("metrics.json", 256 * 1024 + 1), ("uncommitted.patch", 256 * 1024 + 1)],
)
def test_registration_rejects_oversize_before_copying(
    tmp_path: Path, name: str, size: int
) -> None:
    repro = tmp_path / "source"
    write(repro / "run.json", b'{"name":"fixture"}')
    location = repro / "predictions" if name == "metrics.json" else repro
    write(location / name, b" " * size)
    base = tmp_path / "knowledge"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "kg_register.py"),
            "--task",
            "plcs",
            "--repro-dir",
            str(repro),
        ],
        cwd=ROOT,
        env={**os.environ, "KNOWLEDGE_DIR": str(base)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "rejected before copying" in result.stderr
    assert not (base / "runs").exists() and not list(base.rglob(".sequence"))


def test_removed_path_export_validates_records_and_skips_only_pinned_inventory(
    tmp_path: Path,
    artifacts: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "knowledge"
    path = deletion(base)
    monkeypatch.setattr(artifacts, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(artifacts, "nodes_dir", lambda: base / "nodes")
    monkeypatch.setattr(sys, "argv", ["kg_artifacts.py", "--removed-paths"])
    assert artifacts.main() == 0
    assert (
        capsys.readouterr().out == "knowledge/runs/run-fixture/predictions/pred.npz\n"
    )
    data = json.loads(path.read_text())
    data["schema_version"] = 2
    save(path, data)
    with pytest.raises(ValueError, match="unsupported schema_version"):
        artifacts.main()
