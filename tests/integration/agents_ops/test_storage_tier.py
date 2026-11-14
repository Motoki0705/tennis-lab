"""Real file contracts for HDD authority, idle eviction and reversible hydration."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@dataclass
class Lab:
    module: Any
    filesystem: Any
    layout: Any
    usage: Any
    config: Any
    engine: Any
    proc: Path
    now: float

    def unit(self, path: str = "data/task/v1") -> Any:
        directory = self.config.repo / path
        (directory / "nested").mkdir(parents=True)
        (directory / "a.bin").write_bytes(b"apple\n")
        (directory / "nested/b.bin").write_bytes(b"\x00\x01\x04\x09")
        unit = self.layout.Unit(
            path,
            "checkpoint"
            if path == "ckpt"
            else "output"
            if "outputs/" in path
            else "dataset",
        )
        self.age(unit, 15)
        return unit

    def age(self, unit: Any, days: float) -> None:
        for name, entry in self.filesystem.snapshot(
            self.config.repo / unit.path
        ).entries.items():
            if entry["kind"] == "file":
                path = self.config.repo / unit.path / name
                os.utime(
                    path,
                    ns=(
                        int((self.now - days * 86400) * 10**9),
                        path.stat().st_mtime_ns,
                    ),
                )

    def run(
        self, command: str, unit: Any, *, dry_run: bool = False
    ) -> tuple[int, dict[str, Any]]:
        return cast(
            tuple[int, dict[str, Any]],
            self.engine.run(command, [unit.path], dry_run=dry_run),
        )

    def mirror(self, unit: Any) -> None:
        code, report = self.run("mirror", unit)
        assert code == 0, report
        assert report["results"][0]["action"] in {"mirrored", "verified", "unchanged"}

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.filesystem.read_bytes(
                self.config.state / "audit.jsonl"
            ).splitlines()
        ]


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    monkeypatch.syspath_prepend(str(OPS))
    module = importlib.import_module("storage_tier.engine")
    filesystem = importlib.import_module("storage_tier.filesystem")
    layout = importlib.import_module("storage_tier.layout")
    usage = importlib.import_module("storage_tier.usage")
    repo, mount, proc = tmp_path / "D/repo", tmp_path / "E", tmp_path / "proc"
    repo.mkdir(parents=True)
    mount.mkdir()
    proc.mkdir()
    for state in ("jobs", "running"):
        (repo / ".training_queue" / state).mkdir(parents=True)
    config = layout.Config(repo, mount, ("data/task",), settle_seconds=3600)
    config.mirror.mkdir(parents=True)
    # Pass the settling interval without putting aged file atimes in the future:
    # a normal read must update them, so losing O_NOATIME remains detectable.
    result = Lab(
        module, filesystem, layout, usage, config, None, proc, time.time() + 2 * 3600
    )
    result.engine = module.Engine(
        config,
        inspector=usage.Inspector(repo / ".training_queue", proc),
        mount_check=lambda config: None,
        clock=lambda: result.now,
    )
    return result


def test_mirror_evict_hydrate_verifies_known_bytes_and_keeps_ckpt(lab: Lab) -> None:
    unit, checkpoint = lab.unit(), lab.unit("ckpt")
    lab.mirror(unit)
    lab.mirror(checkpoint)
    manifest = lab.engine.manifest(unit)
    assert (
        manifest["local"]["a.bin"]["sha256"] == hashlib.sha256(b"apple\n").hexdigest()
    )
    assert (
        manifest["mirror"]["nested/b.bin"]["sha256"]
        == hashlib.sha256(b"\x00\x01\x04\x09").hexdigest()
    )
    assert lab.run("evict", checkpoint)[1]["results"][0]["action"] == "kept"
    code, report = lab.run("evict", unit)
    assert code == 0, report
    assert report["results"][0]["action"] == "evicted"
    assert not (lab.config.repo / unit.path).exists()
    assert (lab.config.repo / "ckpt/a.bin").exists()
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin") == b"apple\n"
    )
    deletion = next(record for record in lab.records() if record["event"] == "deleted")
    assert deletion["unit"] == unit.path and deletion["logical_bytes"] == 10
    assert deletion["verification"] == "D-sha256+E-size-mtime-ctime"
    assert deletion["time"] and deletion["manifest_verified_at"]
    code, report = lab.run("hydrate", unit)
    assert code == 0, report
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin") == b"apple\n"
    )
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "nested/b.bin")
        == b"\x00\x01\x04\x09"
    )
    assert lab.run("verify", unit)[0] == 0
    assert lab.run("evict", unit)[1]["results"][0]["action"] == "kept"


@pytest.mark.parametrize("operation", ["mirror", "evict", "hydrate", "cycle"])
def test_missing_mount_fails_before_any_storage_write(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    unit = lab.unit()
    before = lab.filesystem.snapshot(lab.config.repo, hashes=True)
    monkeypatch.setattr(
        lab.module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    lab.engine.mount_check = lab.module.mounted
    with pytest.raises(lab.filesystem.StorageError, match="not mounted"):
        lab.run(operation, unit)
    assert (
        before.entries == lab.filesystem.snapshot(lab.config.repo, hashes=True).entries
    )
    assert not lab.config.state.exists()
    assert not (lab.config.mirror / unit.path).exists()


@pytest.mark.parametrize(
    "change", ["contents", "size", "mtime", "ctime", "extra_file", "missing_file"]
)
def test_changed_mirror_is_never_evicted_or_silently_recertified(
    lab: Lab, change: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = lab.config.mirror / unit.path / "a.bin"
    original = path.stat()
    if change == "contents":
        path.write_bytes(b"peach\n")
    elif change == "size":
        path.write_bytes(b"longer than the recorded mirror")
    elif change == "mtime":
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 5000))
    elif change == "ctime":
        path.chmod(0o600 if (original.st_mode & 0o777) != 0o600 else 0o640)
        path.chmod(original.st_mode & 0o777)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert (
            path.stat().st_size == original.st_size
            and path.stat().st_mtime_ns == original.st_mtime_ns
        )
    elif change == "extra_file":
        path.with_name("extra.bin").write_bytes(b"x")
    else:
        path.unlink()
    for operation in ("evict", "cycle", "mirror"):
        code, report = lab.run(operation, unit)
        assert code == 1, report
        assert "E:" in report["results"][0]["error"]
        assert (lab.config.repo / unit.path / "a.bin").exists()
    assert not any(record["event"] == "deleted" for record in lab.records())


def test_expired_corrupt_d_copy_is_not_mirrored_over_its_prior_evidence(
    lab: Lab,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.config.repo / unit.path / "a.bin").write_bytes(b"peach\n")
    lab.age(unit, 15)
    for operation in ("evict", "cycle"):
        code, report = lab.run(operation, unit)
        assert code == 1 and "SHA-256" in report["results"][0]["error"]
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin") == b"apple\n"
    )


@pytest.mark.parametrize(
    "age_days,expected", [(13.999, "kept"), (14, "evicted"), (14.001, "evicted")]
)
def test_idle_boundary_uses_most_recent_file_atime(
    lab: Lab, age_days: float, expected: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = lab.config.repo / unit.path / "nested/b.bin"
    os.utime(
        path, ns=(int((lab.now - age_days * 86400) * 10**9), path.stat().st_mtime_ns)
    )
    code, report = lab.run("evict", unit)
    assert code == 0, report
    assert report["results"][0]["action"] == expected


@pytest.mark.parametrize("state", ["running", "jobs"])
def test_queue_jobs_with_indirect_inputs_keep_all_units(lab: Lab, state: str) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.config.repo / ".training_queue" / state / "training.job").write_text(
        "cd /elsewhere\npython -m training --config hidden-paths.yaml\n"
    )
    code, report = lab.run("evict", unit)
    assert code == 0 and "queue-" in report["results"][0]["reason"]
    assert (lab.config.repo / unit.path).exists()


def test_real_open_descriptor_reached_through_external_hardlink_blocks(
    lab: Lab, tmp_path: Path
) -> None:
    unit = lab.unit()
    alias = tmp_path / "another-name.bin"
    os.link(lab.config.repo / unit.path / "a.bin", alias)
    lab.mirror(unit)
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; f=open(sys.argv[1], 'rb'); print('ready', flush=True); sys.stdin.readline()",
            str(alias),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"
        (lab.proc / str(child.pid)).symlink_to(
            f"/proc/{child.pid}", target_is_directory=True
        )
        code, report = lab.run("evict", unit)
        assert code == 0 and "open-fd" in report["results"][0]["reason"]
        assert (lab.config.repo / unit.path).exists()
    finally:
        assert child.stdin is not None
        child.communicate("finish\n", timeout=5)


def test_mapping_without_fd_and_uninspectable_process_fail_closed(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    process = lab.proc / "123456789"
    (process / "fd").mkdir(parents=True)
    (process / "cwd").symlink_to(lab.config.repo)
    (process / "root").symlink_to("/")
    info = (lab.config.repo / unit.path / "a.bin").stat()
    (process / "maps").write_text(
        f"00001000-00002000 r--p 00000000 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x} {info.st_ino} /external/alias.bin\n"
    )
    code, report = lab.run("evict", unit)
    assert code == 0 and "open-mmap" in report["results"][0]["reason"]
    original = lab.usage.read_bytes

    def denied(path: Path) -> bytes:
        if path == process / "maps":
            raise PermissionError("injected unreadable process")
        return cast(bytes, original(path))

    monkeypatch.setattr(lab.usage, "read_bytes", denied)
    code, report = lab.run("evict", unit)
    assert code == 1 and "usage cannot be verified" in report["results"][0]["error"]
    assert (lab.config.repo / unit.path).exists()


def test_inspection_includes_a_descriptor_in_the_calling_process(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.proc / str(os.getpid())).symlink_to(
        f"/proc/{os.getpid()}", target_is_directory=True
    )
    with (lab.config.repo / unit.path / "a.bin").open("rb"):
        code, report = lab.run("evict", unit)
        assert code == 0 and "open-fd" in report["results"][0]["reason"]
    assert (lab.config.repo / unit.path).exists()


def test_missing_queue_directory_is_not_an_empty_queue(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.config.repo / ".training_queue/running").rmdir()
    code, report = lab.run("evict", unit)
    assert code == 1 and "queue inspection failed" in report["results"][0]["error"]
    assert (lab.config.repo / unit.path).exists()


@pytest.mark.parametrize(
    "operation", ["mirror", "verify", "evict", "index", "hydrate", "cycle"]
)
def test_dry_run_changes_neither_data_nor_metadata_nor_state(
    lab: Lab, tmp_path: Path, operation: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    if operation == "hydrate":
        assert lab.run("evict", unit)[0] == 0
    before = lab.filesystem.snapshot(tmp_path, hashes=True)
    code, report = lab.run(operation, unit, dry_run=True)
    assert code == 0, report
    assert before.entries == lab.filesystem.snapshot(tmp_path, hashes=True).entries


def test_dry_run_of_new_mirror_creates_nothing(lab: Lab, tmp_path: Path) -> None:
    unit = lab.unit()
    before = lab.filesystem.snapshot(tmp_path, hashes=True)
    code, report = lab.run("cycle", unit, dry_run=True)
    assert code == 0 and report["results"][0]["action"] == "would_mirror"
    assert before.entries == lab.filesystem.snapshot(tmp_path, hashes=True).entries
    assert not lab.config.state.exists()


def test_archive_only_index_then_hydrate_initializes_both_sides(lab: Lab) -> None:
    unit = lab.unit()
    archive = lab.config.mirror / unit.path
    archive.parent.mkdir(parents=True)
    os.rename(lab.config.repo / unit.path, archive)
    code, report = lab.run("index", unit, dry_run=True)
    assert code == 0 and report["results"][0]["atime_preserved"]
    assert not lab.engine.manifest_path(unit).exists()
    assert lab.run("index", unit)[0] == 0
    assert lab.engine.manifest(unit)["local"] is None
    assert lab.run("hydrate", unit)[0] == 0
    assert lab.run("verify", unit)[0] == 0


def test_unrecorded_conflicting_e_copy_and_existing_hydrate_target_are_preserved(
    lab: Lab,
) -> None:
    unit = lab.unit()
    destination = lab.config.mirror / unit.path
    (destination / "nested").mkdir(parents=True)
    (destination / "a.bin").write_bytes(b"peach\n")
    (destination / "nested/b.bin").write_bytes(b"\x00\x01\x04\x09")
    code, report = lab.run("mirror", unit)
    assert code == 1 and "refusing to overwrite" in report["results"][0]["error"]
    assert lab.filesystem.read_bytes(destination / "a.bin") == b"peach\n"
    code, report = lab.run("hydrate", unit)
    assert code == 1 and "already exists" in report["results"][0]["error"]
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin") == b"apple\n"
    )


def test_mirror_can_update_a_recorded_quiescent_warm_unit(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.config.repo / unit.path / "a.bin").write_bytes(b"new output\n")
    lab.age(unit, 1)
    code, report = lab.run("cycle", unit)
    assert code == 0, report
    assert report["results"][0]["action"] == "mirrored"
    assert report["results"][0]["eviction"]["action"] == "kept"
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin")
        == b"new output\n"
    )


def test_cross_unit_hardlinks_keep_other_reference_and_report_unreclaimed_bytes(
    lab: Lab, tmp_path: Path
) -> None:
    unit = lab.unit()
    source = lab.config.repo / unit.path
    alias = tmp_path / "other-output.bin"
    os.link(source / "a.bin", alias)
    os.link(source / "nested/b.bin", source / "duplicate.bin")
    lab.mirror(unit)
    mirror = lab.config.mirror / unit.path
    assert (mirror / "nested/b.bin").stat().st_ino == (
        mirror / "duplicate.bin"
    ).stat().st_ino
    code, report = lab.run("evict", unit)
    assert code == 0, report
    result = report["results"][0]
    assert result["logical_bytes"] == 14
    assert (
        result["external_hardlink_inodes"] == 1
        and result["external_hardlink_bytes"] == 6
    )
    assert (
        result["estimated_reclaim_bytes"]
        == 512 * (mirror / "nested/b.bin").stat().st_blocks
    )
    assert lab.filesystem.read_bytes(alias) == b"apple\n"
    assert lab.run("hydrate", unit)[0] == 0
    assert (source / "nested/b.bin").stat().st_ino == (
        source / "duplicate.bin"
    ).stat().st_ino
    assert (source / "a.bin").stat().st_ino != alias.stat().st_ino


def test_symlink_unit_is_refused_without_changing_its_atime(
    lab: Lab, tmp_path: Path
) -> None:
    unit = lab.unit()
    source = lab.config.repo / unit.path
    (source / "alias.bin").symlink_to("nested/b.bin")
    (source / "external.bin").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_bytes(b"never delete")
    old = int((lab.now - 20 * 86400) * 10**9)
    for name in ("alias.bin", "external.bin"):
        os.utime(source / name, ns=(old, old), follow_symlinks=False)
    before = {n: (source / n).lstat() for n in ("alias.bin", "external.bin")}
    with pytest.raises(lab.filesystem.StorageError, match="symlink"):
        lab.filesystem.snapshot(source, hashes=True)
    assert before == {n: (source / n).lstat() for n in before}
    assert lab.filesystem.read_bytes(tmp_path / "outside") == b"never delete"


def test_write_or_read_during_verification_prevents_eviction(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.age(unit, 15)
    lab.mirror(unit)
    original = lab.module.snapshot
    accessed = False

    def concurrent_read(path: Path, *, hashes: bool = False) -> Any:
        nonlocal accessed
        result = original(path, hashes=hashes)
        if path == lab.config.repo / unit.path and hashes and not accessed:
            accessed = True
            with (path / "a.bin").open("rb") as stream:
                assert stream.read() == b"apple\n"
        return result

    monkeypatch.setattr(lab.module, "snapshot", concurrent_read)
    code, report = lab.run("evict", unit)
    assert accessed and code == 0, report
    assert report["results"][0]["action"] == "kept"
    assert (lab.config.repo / unit.path).exists()


def test_source_write_during_copy_does_not_publish_e_copy(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    original = lab.module.copy_tree

    def changed(source: Path, target: Path, expected: Any, **kwargs: Any) -> None:
        original(source, target, expected, **kwargs)
        (source / "a.bin").write_bytes(b"written during copy")

    monkeypatch.setattr(lab.module, "copy_tree", changed)
    code, report = lab.run("mirror", unit)
    assert code == 1 and "stability" in report["results"][0]["error"]
    assert not (lab.config.mirror / unit.path).exists()
    assert not lab.engine.manifest_path(unit).exists()
    assert not list(
        (lab.config.mirror / unit.path).parent.glob(".storage-tier-stage-*")
    )


def test_copy_corruption_is_detected_by_actual_e_hashes(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    original = lab.module.copy_tree

    def corrupt(source: Path, target: Path, expected: Any, **kwargs: Any) -> None:
        original(source, target, expected, **kwargs)
        (target / "a.bin").write_bytes(b"peach\n")

    monkeypatch.setattr(lab.module, "copy_tree", corrupt)
    code, report = lab.run("mirror", unit)
    assert code == 1 and "SHA-256" in report["results"][0]["error"]
    assert not (lab.config.mirror / unit.path).exists()
    assert not lab.engine.manifest_path(unit).exists()


def test_hydrate_checks_e_hash_against_existing_manifest(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    assert lab.run("evict", unit)[0] == 0
    manifest = lab.engine.manifest(unit)
    manifest["mirror"]["a.bin"]["sha256"] = hashlib.sha256(b"peach\n").hexdigest()
    lab.filesystem.write_json(lab.engine.manifest_path(unit), manifest)
    code, report = lab.run("hydrate", unit)
    assert code == 1 and "SHA-256" in report["results"][0]["error"]
    assert not (lab.config.repo / unit.path).exists()


def test_mount_loss_after_quarantine_restores_source_without_deletion(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    calls = 0

    def lost(config: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise lab.filesystem.StorageError("E: unmounted after quarantine")

    lab.engine.mount_check = lost
    code, report = lab.run("evict", unit)
    assert calls == 4 and code == 1 and "unmounted" in report["results"][0]["error"]
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin") == b"apple\n"
    )
    assert not any(record["event"] == "deleted" for record in lab.records())


@pytest.mark.parametrize("operation", ["mirror", "hydrate"])
def test_publication_writer_cannot_attach_stale_hashes_to_fresh_metadata(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    local, mirror = lab.config.repo / unit.path, lab.config.mirror / unit.path
    if operation == "hydrate":
        assert lab.run("evict", unit)[0] == 0
        destination = local
    else:
        (local / "a.bin").write_bytes(b"peach\n")
        lab.age(unit, 1)
        destination = mirror
    manifest_before = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    original = lab.module.rename
    published = False

    def writer(source: Path, target: Path, *, exchange: bool = False) -> None:
        nonlocal published
        original(source, target, exchange=exchange)
        if target == destination:
            published = True
            (target / "a.bin").write_bytes(b"grape\n")

    monkeypatch.setattr(lab.module, "rename", writer)
    code, report = lab.run(operation, unit)
    assert published and code == 1, report
    assert "publication" in report["results"][0]["error"]
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == manifest_before
    if operation == "mirror":
        assert lab.filesystem.read_bytes(local / "a.bin") == b"peach\n"
        backups = list(mirror.parent.glob(".storage-tier-stage-*"))
        assert len(backups) == 1
        assert lab.filesystem.read_bytes(backups[0] / "a.bin") == b"apple\n"
    else:
        assert lab.filesystem.read_bytes(mirror / "a.bin") == b"apple\n"


def test_recent_writer_and_empty_unit_are_retained(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    lab.now = time.time()
    lab.age(unit, 15)
    code, report = lab.run("evict", unit)
    assert code == 0 and "recent writes" in report["results"][0]["reason"]
    empty = lab.layout.Unit("data/task/empty", "dataset")
    (lab.config.repo / empty.path).mkdir()
    code, report = lab.run("evict", empty)
    assert code == 0 and "no regular-file atime" in report["results"][0]["reason"]


def test_interrupted_eviction_restores_then_requires_recovery(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    original = lab.module.remove_tree

    def interrupted(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path).name.startswith(".storage-tier-trash-"):
            raise OSError("injected deletion interruption")
        original(path, *args, **kwargs)

    monkeypatch.setattr(lab.module, "remove_tree", interrupted)
    code, report = lab.run("evict", unit)
    assert code == 1 and "interruption" in report["results"][0]["error"]
    assert (lab.config.repo / unit.path / "a.bin").exists()
    with pytest.raises(lab.filesystem.StorageError, match="transactions"):
        lab.run("evict", unit)
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and report["results"][0]["partial_deletion_possible"]
    monkeypatch.setattr(lab.module, "remove_tree", original)
    assert lab.run("evict", unit)[0] == 0


def test_successor_created_before_quarantine_is_restored_without_deletion(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    saved = source.with_name("saved-original")
    original = lab.module.rename
    replaced = False

    def race(first: Path, second: Path, *, exchange: bool = False) -> None:
        nonlocal replaced
        if first == source and not replaced:
            replaced = True
            os.rename(source, saved)
            source.mkdir()
            (source / "successor.bin").write_bytes(b"new owner's data")
        original(first, second, exchange=exchange)

    monkeypatch.setattr(lab.module, "rename", race)
    code, report = lab.run("evict", unit)
    assert code == 1 and "replaced" in report["results"][0]["error"]
    assert lab.filesystem.read_bytes(source / "successor.bin") == b"new owner's data"
    assert lab.filesystem.read_bytes(saved / "a.bin") == b"apple\n"
    assert not any(record["event"] == "deleted" for record in lab.records())


def test_discovery_uses_explicit_dataset_boundaries_and_complete_run_layout(
    lab: Lab,
) -> None:
    from src.utils.configuration.output_layout import OUTPUT_KINDS

    assert lab.layout.PURPOSES == OUTPUT_KINDS
    version = lab.unit()
    run = lab.unit("outputs/task/train/experiment/run-001")
    worktree_run = lab.unit(
        ".claude/worktrees/worker/outputs/task/evaluate/experiment/run-002"
    )
    legacy = lab.config.repo / "outputs/task/train/old-run"
    (legacy / "logs/version_0").mkdir(parents=True)
    (legacy / "config.yaml").write_text("legacy: true\n")
    raw = lab.config.repo / "data/raw-assets/parts"
    raw.mkdir(parents=True)
    (lab.config.repo / "outputs/task/train/space name/run").mkdir(parents=True)
    inventory = lab.layout.discover(lab.config, lab.config.repo)
    assert inventory.units == sorted([version, run, worktree_run])
    assert "outputs/task/train/old-run" in inventory.unmanaged
    assert "data/raw-assets" in inventory.unmanaged
    assert "outputs/task/train/space name" in inventory.unmanaged
    (lab.config.repo / "data").rename(lab.config.repo / "old-data")
    (lab.config.repo / "data").symlink_to(
        lab.config.repo / "old-data", target_is_directory=True
    )
    os.utime(lab.config.repo / "data", ns=(1, 1), follow_symlinks=False)
    link_before = (lab.config.repo / "data").lstat()
    inventory = lab.layout.discover(lab.config, lab.config.repo)
    assert inventory.errors and version not in inventory.units
    assert (lab.config.repo / "data").lstat() == link_before


def test_same_device_bind_mount_is_rejected_before_copying(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    original = lab.filesystem.fd_mount_id
    nested = lab.config.repo / unit.path / "nested"

    def other_mount(fd: int) -> int:
        value = cast(int, original(fd))
        return (
            value + 1 if Path(os.readlink(f"/proc/self/fd/{fd}")) == nested else value
        )

    monkeypatch.setattr(lab.filesystem, "fd_mount_id", other_mount)
    code, report = lab.run("mirror", unit)
    assert code == 1 and "nested mount" in report["results"][0]["error"]
    assert not (lab.config.mirror / unit.path).exists()
    assert lab.filesystem.read_bytes(nested / "b.bin") == b"\x00\x01\x04\x09"


def test_replacement_after_final_inspection_is_never_deleted_or_restored_as_ours(
    lab: Lab,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    saved = source.with_name("saved-verified-root")
    replacements: list[Path] = []
    calls = 0

    def replace_after_inspection(config: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            (quarantine,) = source.parent.glob(".storage-tier-trash-*")
            quarantine.rename(saved)
            quarantine.mkdir()
            (quarantine / "successor.bin").write_bytes(b"new data without a mirror")
            replacements.append(quarantine)

    lab.engine.mount_check = replace_after_inspection
    code, report = lab.run("evict", unit)
    assert calls == 4 and code == 1, report
    assert "identity" in report["results"][0]["error"]
    assert (
        lab.filesystem.read_bytes(replacements[0] / "successor.bin")
        == b"new data without a mirror"
    )
    assert lab.filesystem.read_bytes(saved / "a.bin") == b"apple\n"
    with pytest.raises(lab.filesystem.StorageError, match="identity"):
        lab.engine.run("recover", [], dry_run=False)
    assert (replacements[0] / "successor.bin").exists()
    assert not any(record["event"] == "deleted" for record in lab.records())


def test_o_noatime_failure_never_retries_with_an_ordinary_open(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    original = os.open
    attempts = []

    def denied(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if str(path) == "a.bin" and kwargs.get("dir_fd") is not None:
            attempts.append(flags)
            raise PermissionError("O_NOATIME denied")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied)
    code, report = lab.run("mirror", unit)
    assert code == 1 and "O_NOATIME denied" in report["results"][0]["error"]
    assert len(attempts) == 1 and attempts[0] & os.O_NOATIME
    assert (lab.config.repo / unit.path).exists() and not (
        lab.config.mirror / unit.path
    ).exists()
