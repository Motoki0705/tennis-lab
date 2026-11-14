"""Real file contracts for HDD authority, idle eviction and reversible hydration."""

from __future__ import annotations

import errno
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
    config = layout.Config(repo, mount, ("data/task",))
    config.mirror.mkdir(parents=True)
    # Aged atimes are truly in the past: an ordinary read must update them.
    result = Lab(module, filesystem, layout, usage, config, None, proc, time.time())
    result.engine = module.Engine(
        config,
        inspector=usage.Inspector(proc),
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
def test_queue_records_do_not_protect_idle_units(lab: Lab, state: str) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    directory = lab.config.repo / ".training_queue" / state
    directory.mkdir(parents=True)
    (directory / "training.job").write_text(
        "cd /elsewhere\npython -m training --config hidden-paths.yaml\n"
    )
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted"
    assert not (lab.config.repo / unit.path).exists()


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


@pytest.mark.parametrize("denied_entry", ["fd", "maps"])
def test_mapping_without_fd_and_uninspectable_process_fail_closed(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, denied_entry: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    process = lab.proc / "123456789"
    (process / "fd").mkdir(parents=True)
    (process / "status").write_text(f"Uid:\t0\t{os.geteuid()}\t0\t0\n")
    (process / "cwd").symlink_to(lab.config.repo)
    (process / "root").symlink_to("/")
    info = (lab.config.repo / unit.path / "a.bin").stat()
    (process / "maps").write_text(
        f"00001000-00002000 r--p 00000000 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x} {info.st_ino} /external/alias.bin\n"
    )
    code, report = lab.run("evict", unit)
    assert code == 0 and "open-mmap" in report["results"][0]["reason"]
    original = lab.usage.read_bytes
    original_names = lab.usage.names

    def denied(path: Path) -> bytes:
        if denied_entry == "maps" and path == process / "maps":
            raise PermissionError("injected unreadable process")
        return cast(bytes, original(path))

    monkeypatch.setattr(lab.usage, "read_bytes", denied)

    def denied_names(path: Path) -> list[str]:
        if denied_entry == "fd" and path == process / "fd":
            raise PermissionError("injected unreadable process")
        return cast(list[str], original_names(path))

    monkeypatch.setattr(lab.usage, "names", denied_names)
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


def test_no_queue_directory_is_needed_for_eviction(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    assert not (lab.config.repo / ".training_queue").exists()
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted"


def test_other_uid_is_excluded_before_fd_maps_and_is_audited(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    process = lab.proc / "123456789"
    process.mkdir()
    other_uid = os.geteuid() + 12345
    # An uninspectable fd inventory and absent maps must never be inspected.
    (process / "status").write_text(f"Uid:\t{os.geteuid()}\t{other_uid}\t0\t0\n")
    (process / "fd").write_text("not a readable descriptor directory")
    code, preview = lab.run("cycle", unit, dry_run=True)
    assert code == 0 and preview["results"][0]["action"] == "would_evict", preview
    assert preview["usage"]["unknown"] == []
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted", report
    scope = report["process_scope"]
    assert scope["uid"] == os.geteuid()
    assert scope["excluded_other_uid"] == 1
    assert scope["excluded_uids"] == {str(other_uid): 1}
    deletion = next(row for row in lab.records() if row["event"] == "deleted")
    assert deletion["process_scope"] == scope


@pytest.fixture
def private_process(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Synthetic kernel metadata; filesystem ownership/errors are independent inputs."""
    process = lab.proc / "123456789"
    process.mkdir()
    (process / "fd").mkdir()
    (process / "maps").write_text("")
    (process / "cwd").symlink_to(lab.config.repo)
    (process / "root").symlink_to("/")
    uid = 12345
    lab.engine.inspector = lab.usage.Inspector(lab.proc, uid=uid)
    (process / "status").write_text(
        f"Name:\tprivate-agent\nState:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nCoreDumping:\t0\n"
    )
    fields = ["S", *("0" for _ in range(18)), "314159"]
    (process / "stat").write_text(f"{process.name} (private agent) " + " ".join(fields))
    options: dict[str, Any] = {
        "path": process,
        "uid": uid,
        "owner": 0,
        "denied": {"fd", "maps", "cwd"},
        "errno": errno.EACCES,
        "reuse": False,
    }
    original_names, original_read = lab.usage.names, lab.usage.read_bytes
    original_link, original_stat = os.readlink, Path.lstat
    stat_reads = 0

    def check(path: Path, name: str) -> None:
        if path == process / name and name in options["denied"]:
            raise OSError(options["errno"], "injected kernel read denial", str(path))

    def directory(path: Path) -> list[str]:
        check(path, "fd")
        return cast(list[str], original_names(path))

    def read(path: Path) -> bytes:
        nonlocal stat_reads
        check(path, "maps")
        value = cast(bytes, original_read(path))
        if path == process / "stat":
            stat_reads += 1
            if options["reuse"]:
                return value.replace(b"314159", str(314159 + stat_reads).encode())
        return value

    def link(path: Any, *args: Any, **kwargs: Any) -> Any:
        check(Path(path), "cwd")
        return original_link(path, *args, **kwargs)

    def lstat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        info = original_stat(path, *args, **kwargs)
        if path.parent == process and path.name in {"fd", "maps", "cwd"}:
            values = list(info)
            values[4:6] = [options["owner"], options["owner"]]
            return os.stat_result(values)
        return info

    monkeypatch.setattr(lab.usage, "names", directory)
    monkeypatch.setattr(lab.usage, "read_bytes", read)
    monkeypatch.setattr(os, "readlink", link)
    monkeypatch.setattr(Path, "lstat", lstat)
    return options


def test_nondumpable_same_uid_is_excluded_with_proof_and_audit(
    lab: Lab, private_process: dict[str, Any]
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    code, report = lab.run("cycle", unit, dry_run=True)
    assert code == 0 and report["usage"]["unknown"] == [], report
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted", report
    scope = report["process_scope"]
    assert scope["excluded_nondumpable_count"] == 1
    (excluded,) = scope["excluded_nondumpable"]
    assert excluded["comm"] == "private-agent" and excluded["pid"] == 123456789
    assert excluded["denied"] == {"fd": "EACCES", "maps": "EACCES", "cwd": "EACCES"}
    assert excluded["starttime_ticks"] == 314159
    assert (
        next(row for row in lab.records() if row["event"] == "deleted")["process_scope"]
        == scope
    )


@pytest.mark.parametrize(
    "failure", ["dumpable", "user-owned", "partial", "io-error", "pid-reused"]
)
def test_denied_same_uid_without_nondumpable_evidence_fails_closed(
    lab: Lab, private_process: dict[str, Any], failure: str
) -> None:
    process = private_process["path"]
    unit = lab.unit()
    # Establish the mirror before introducing an inspection failure.
    private_process["denied"] = set()
    lab.mirror(unit)
    private_process["denied"] = {"fd", "maps", "cwd"}
    if failure == "dumpable":
        with (process / "status").open("a") as stream:
            stream.write("Dumpable:\t1\n")
    elif failure == "user-owned":
        private_process["owner"] = private_process["uid"]
    elif failure == "partial":
        private_process["denied"] = {"maps", "cwd"}
    elif failure == "io-error":
        private_process["errno"] = errno.EIO
    else:
        private_process["reuse"] = True
    code, report = lab.run("evict", unit)
    assert code == 1 and report["results"][0]["action"] == "failed", report
    assert report["process_scope"]["excluded_nondumpable_count"] == 0
    assert (lab.config.repo / unit.path / "a.bin").exists()
    if failure == "pid-reused":
        # The same identity churn must not authorize an exited-process exemption.
        path = process / "status"
        path.write_text(
            path.read_text().replace("State:\tS (sleeping)", "State:\tZ (zombie)")
            + "Threads:\t1\n"
        )
        path = process / "stat"
        path.write_text(path.read_text().replace(") S ", ") Z "))
        (process / "task" / process.name).mkdir(parents=True)
        code, report = lab.run("evict", unit)
        assert code == 1 and report["process_scope"]["exited_processes"] == 0, report
        assert (lab.config.repo / unit.path / "a.bin").exists()


@pytest.fixture
def capability_process(lab: Lab, private_process: dict[str, Any]) -> dict[str, Any]:
    process, uid = private_process["path"], private_process["uid"]
    private_process["owner"] = uid
    private_process["denied"] = {"maps", "cwd"}
    with (process / "status").open("a") as stream:
        stream.write("Dumpable:\t1\nCapPrm:\t0000000800000000\n")
    caller = lab.proc / "self"
    caller.mkdir()
    (caller / "status").write_text(
        f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nCapPrm:\t0000000000000002\n"
    )
    # Formatting is irrelevant; equal mappings still do not prove identity.
    # No ns/user handle exists: exclusion must not try to open one.
    (caller / "uid_map").write_text("0 0 4294967295\n")
    (process / "uid_map").write_text("         0          0 4294967295\n")
    return private_process


@pytest.mark.parametrize(
    "auxiliary", ["matched", "target-denied", "inspector-io-error", "target-missing"]
)
def test_capability_difference_audits_namespace_assumption_and_uid_maps(
    lab: Lab,
    capability_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    auxiliary: str,
) -> None:
    process = capability_process["path"]
    caller = lab.proc / "self"
    if auxiliary == "target-missing":
        (process / "uid_map").unlink()
    elif auxiliary in {"target-denied", "inspector-io-error"}:
        path = (process if auxiliary == "target-denied" else caller) / "uid_map"
        error = errno.EACCES if auxiliary == "target-denied" else errno.EIO
        original_read = lab.usage.read_bytes

        def read(candidate: Path) -> bytes:
            if candidate == path:
                raise OSError(error, "injected auxiliary uid_map failure", str(path))
            return cast(bytes, original_read(candidate))

        monkeypatch.setattr(lab.usage, "read_bytes", read)
    unit = lab.unit()
    lab.mirror(unit)
    code, preview = lab.run("cycle", unit, dry_run=True)
    assert code == 0 and preview["usage"]["unknown"] == [], preview
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted", report
    scope = report["process_scope"]
    assert scope["excluded_capabilities_count"] == 1
    assert scope["excluded_nondumpable_count"] == 0
    (excluded,) = scope["excluded_capabilities"]
    assert excluded["kind"] == "permitted-capabilities"
    assert excluded["comm"] == "private-agent" and excluded["pid"] == 123456789
    assert excluded["target_cap_prm"] == "0000000800000000"
    assert excluded["inspector_cap_prm"] == "0000000000000002"
    assert excluded["denied"] == {"maps": "EACCES", "cwd": "EACCES"}
    namespace = excluded["user_namespace"]
    assert namespace["identity"] == "unverified"
    assert namespace["assumption"] == "same-user-namespace"
    for key in ("uid_map_before", "uid_map_after"):
        observation = namespace[key]
        if auxiliary == "matched":
            assert observation["comparison"] == "matched"
            assert (
                observation["target"]
                == observation["inspector"]
                == {"ranges": [[0, 0, 4294967295]]}
            )
        else:
            assert observation["comparison"] == "unverified"
            failed = "inspector" if auxiliary == "inspector-io-error" else "target"
            expected = {
                "target-denied": "EACCES",
                "inspector-io-error": "EIO",
                "target-missing": "ENOENT",
            }[auxiliary]
            assert observation[failed]["errno"] == expected
            assert "error" in observation[failed]
    assert (
        next(row for row in lab.records() if row["event"] == "deleted")["process_scope"]
        == scope
    )


@pytest.mark.parametrize(
    "failure",
    [
        "subset",
        "different-uid-map",
        "changed-uid-map",
        "malformed-uid-map",
        "malformed-target",
        "missing-caller",
        "partial-denial",
        "pid-reused",
        "caller-changed",
    ],
)
def test_capability_exclusion_requires_complete_stable_evidence(
    lab: Lab,
    capability_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    process = capability_process["path"]
    caller_status = lab.proc / "self/status"
    if failure == "subset":
        # A strict superset, not equality: every target bit is already available.
        caller_status.write_text(
            caller_status.read_text().replace("0000000000000002", "0000000800000002")
        )
    elif failure == "different-uid-map":
        (process / "uid_map").write_text("0 1000 1\n")
    elif failure == "malformed-uid-map":
        (process / "uid_map").write_text("invalid mapping\n")
    elif failure == "malformed-target":
        path = process / "status"
        path.write_text(path.read_text().replace("0000000800000000", "not-hex"))
    elif failure == "missing-caller":
        caller_status.unlink()
    elif failure == "partial-denial":
        capability_process["denied"] = {"maps"}
    elif failure == "pid-reused":
        capability_process["reuse"] = True
    else:
        original_read = lab.usage.read_bytes
        reads = 0

        def read(path: Path) -> bytes:
            nonlocal reads
            value = cast(bytes, original_read(path))
            if path == caller_status and failure == "caller-changed":
                reads += 1
                if reads > 1:
                    return value.replace(b"0000000000000002", b"0000000800000002")
            if path == process / "uid_map" and failure == "changed-uid-map":
                reads += 1
                if reads > 1:
                    return b"0 1000 1\n"
            return value

        monkeypatch.setattr(lab.usage, "read_bytes", read)
    code, report = lab.run("evict", unit)
    assert code == 1 and report["results"][0]["action"] == "failed", report
    assert report["process_scope"]["excluded_capabilities_count"] == 0
    assert report["process_scope"]["excluded_nondumpable_count"] == 0
    assert (lab.config.repo / unit.path / "a.bin").exists()


def test_known_zombie_is_counted_as_exited_without_inspecting_resources(
    lab: Lab,
) -> None:
    process = lab.proc / "123456789"
    process.mkdir()
    (process / "status").write_text(
        f"Name:\tfinished\nState:\tZ (zombie)\nThreads:\t1\nUid:\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\n"
    )
    (process / "task" / process.name).mkdir(parents=True)
    (process / "stat").write_text(
        f"{process.name} (finished) Z " + "0 " * 18 + "271828\n"
    )
    unit = lab.unit()
    code, report = lab.run("cycle", unit, dry_run=True)
    assert code == 0 and report["usage"]["unknown"] == [], report
    assert report["process_scope"]["exited_processes"] == 1
    assert report["process_scope"]["excluded_nondumpable_count"] == 0


@pytest.mark.parametrize(
    "threads,tasks",
    [
        ("2", ["123456789", "123456790"]),
        (None, ["123456789"]),
        ("1", ["123456789", "123456790"]),
    ],
)
def test_zombie_leader_with_live_or_unknown_threads_is_not_exited(
    lab: Lab, threads: str | None, tasks: list[str]
) -> None:
    process = lab.proc / "123456789"
    process.mkdir()
    status = f"State:\tZ (zombie)\nUid:\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\n"
    if threads is not None:
        status += f"Threads:\t{threads}\n"
    (process / "status").write_text(status)
    (process / "stat").write_text(
        f"{process.name} (finished) Z " + "0 " * 18 + "271828\n"
    )
    for task in tasks:
        (process / "task" / task).mkdir(parents=True)
    unit = lab.unit()
    code, report = lab.run("cycle", unit, dry_run=True)
    assert code == 1 and report["usage"]["unknown"], report
    assert report["process_scope"]["exited_processes"] == 0
    assert (lab.config.repo / unit.path / "a.bin").exists()


@pytest.mark.parametrize("dumpable", [0, 1])
def test_native_prctl_dumpability_is_distinguished_by_kernel_evidence(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, dumpable: int
) -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import ctypes,sys; c=ctypes.CDLL(None); assert c.prctl(15,b'storage-probe',0,0,0)==0; assert c.prctl(4,int(sys.argv[1]),0,0,0)==0; print('ready:'+str(c.prctl(3)),flush=True); sys.stdin.readline()",
            str(dumpable),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert (
            child.stdout is not None
            and child.stdout.readline().strip() == f"ready:{dumpable}"
        )
        try:
            (Path("/proc") / str(child.pid) / "maps").read_bytes()
            independently_readable = True
        except PermissionError:
            independently_readable = False
        inspector = lab.usage.Inspector()
        monkeypatch.setattr(inspector, "process_names", lambda: [str(child.pid)])
        result = inspector.inspect((), lab.filesystem.Snapshot({}))
        assert result.unknown == []
        assert result.scope()["excluded_nondumpable_count"] == int(
            not independently_readable
        )
        if not independently_readable:
            assert result.scope()["excluded_nondumpable"][0]["comm"] == "storage-probe"
        else:
            assert result.scope()["inspected_processes"] == 1
    finally:
        assert child.stdin is not None
        child.communicate("done\n", timeout=5)


@pytest.mark.parametrize("failure", ["unreadable", "malformed", "missing"])
def test_unknown_process_uid_is_not_silently_excluded(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    process = lab.proc / "123456789"
    process.mkdir()
    status = process / "status"
    status.write_text("Uid: invalid\n")
    if failure == "missing":
        status.unlink()
    original = lab.usage.read_bytes

    def read(path: Path) -> bytes:
        if path == status and failure == "unreadable":
            raise PermissionError("injected unreadable UID")
        return cast(bytes, original(path))

    monkeypatch.setattr(lab.usage, "read_bytes", read)
    code, report = lab.run("evict", unit)
    assert code == 1 and "usage cannot be verified" in report["results"][0]["error"]
    assert (lab.config.repo / unit.path / "a.bin").exists()


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


def test_recent_ctime_does_not_override_idle_atime_and_empty_unit_is_kept(
    lab: Lab,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    lab.now = time.time()
    lab.age(unit, 15)
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted"
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
    worktree_run = lab.unit(".worktree-outputs/worker/task/evaluate/experiment/run-002")
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


def test_worktree_links_are_references_and_orphaned_outputs_remain_units(
    lab: Lab,
) -> None:
    main = lab.unit("outputs/task/train/experiment/run-001")
    orphan = lab.unit(".worktree-outputs/gone/task/train/experiment/run-002")
    worktree = lab.config.repo / ".claude/worktrees/alias"
    worktree.mkdir(parents=True)
    (worktree / "outputs").symlink_to(lab.config.repo / "outputs")
    os.utime(worktree / "outputs", ns=(1, 1), follow_symlinks=False)
    before = (worktree / "outputs").lstat()
    inventory = lab.layout.discover(lab.config, lab.config.repo)
    assert inventory.units == sorted([main, orphan])
    assert inventory.references == [".claude/worktrees/alias/outputs"]
    assert inventory.errors == []
    assert (worktree / "outputs").lstat() == before
    legacy = lab.config.mirror / ".claude/worktrees/old/outputs"
    legacy.mkdir(parents=True)
    archive = lab.layout.discover(lab.config, lab.config.mirror)
    assert archive.legacy_outputs == [
        {
            "source": ".claude/worktrees/old/outputs",
            "destination": ".worktree-outputs/old",
        }
    ]
    assert archive.units == []


def test_scan_sizes_unclassified_e_data_without_reading_payloads_or_links(
    lab: Lab,
) -> None:
    root = lab.config.mirror / "data/ACCAD"
    root.mkdir(parents=True)
    (root / "motion.bin").write_bytes(b"motion" * 1000)
    os.link(root / "motion.bin", root / "motion-hardlink.bin")
    outside = lab.config.mirror / "outside.bin"
    outside.write_bytes(b"outside" * 2000)
    (root / "outside-link").symlink_to(outside)
    for path in (root, root / "motion.bin", root / "outside-link", outside):
        os.utime(path, ns=(1, path.lstat().st_mtime_ns), follow_symlinks=False)
    before = {
        str(p): p.lstat()
        for p in (root, root / "motion.bin", root / "outside-link", outside)
    }
    expected_allocated = sum(
        (root / name).lstat().st_blocks * 512
        for name in (".", "motion.bin", "outside-link")
    )
    code, report = lab.engine.run("scan", [], dry_run=False)
    assert code == 0, report
    (size,) = report["unmanaged_data_sizes_E"]
    assert size["path"] == "data/ACCAD" and size["logical_bytes"] == 6000
    assert size["allocated_bytes"] == expected_allocated
    assert size["files"] == 2 and size["symlinks"] == 1
    assert before == {name: Path(name).lstat() for name in before}
    assert not lab.config.state.exists()


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
