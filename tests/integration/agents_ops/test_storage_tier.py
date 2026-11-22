"""Real file contracts for HDD authority, idle eviction and reversible hydration."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from tests.support.agents_ops.storage_tier import OPS, Lab, make_lab


def filesystem_path(path: Path) -> str:
    """Independent util-linux inventory for synthetic mount-root test inputs."""
    record = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(path), "--output", "FSROOT,TARGET"],
            text=True,
        )
    )["filesystems"][0]
    return str(Path(record["fsroot"]) / path.relative_to(record["target"]))


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    monkeypatch.syspath_prepend(str(OPS))
    result = make_lab(tmp_path)
    # Synthetic /proc transitions are driven by test inputs, never wall time.
    monkeypatch.setattr(result.usage, "sleep", lambda delay: None)
    return result


def test_mirror_evict_hydrate_verifies_known_bytes_and_keeps_ckpt(lab: Lab) -> None:
    unit, checkpoint = lab.unit(), lab.unit("ckpt/model")
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
    assert (lab.config.repo / "ckpt/model/a.bin").exists()
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


@pytest.mark.parametrize("open_kind", ["read", "O_PATH"])
def test_real_open_descriptor_reached_through_external_hardlink_blocks(
    lab: Lab, tmp_path: Path, open_kind: str
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
            "import os,sys; fd=os.open(sys.argv[1], os.O_PATH if sys.argv[2]=='O_PATH' else os.O_RDONLY); print('ready', flush=True); sys.stdin.readline(); os.close(fd)",
            str(alias),
            open_kind,
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
    (process / "stat").write_text(
        f"{process.name} (mapped) " + " ".join(["S", *(["0"] * 18), "314159"])
    )
    (process / "cwd").symlink_to(lab.config.repo)
    (process / "root").symlink_to("/")
    info = (lab.config.repo / unit.path / "a.bin").stat()
    (process / "maps").write_text(
        f"00001000-00002000 r--p 00000000 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x} {info.st_ino} /external/alias.bin\n"
    )
    (process / "mountinfo").write_text("")
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
    (process / "mountinfo").write_text("")
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


@pytest.mark.parametrize(
    "outcome",
    [
        "gone",
        "unrelated",
        "fd",
        "cwd",
        "root",
        "maps",
        "mount",
        "persistent",
        "replaced",
    ],
)
def test_permission_recheck_pins_identity_and_reinspects_all_references(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    payload = directory / "a.bin"
    process = private_process["path"]
    private_process["denied"] = {"fd"}  # maps is readable: no exclusion proof.
    waits: list[float] = []
    info = payload.stat()
    namespace = tmp_path / "namespace"
    if outcome == "mount":
        namespace.mkdir()
        os.link(payload, namespace / "alias")

    def resume(delay: float) -> None:
        waits.append(delay)
        if outcome == "persistent":
            return
        if outcome == "gone":
            process.rename(process.with_name("finished"))
            return
        if outcome == "replaced":
            path = process / "stat"
            path.write_bytes(path.read_bytes().replace(b"314159", b"314160"))
            return
        private_process["denied"] = set()
        if outcome == "fd":
            (process / "fd/7").symlink_to(payload)
        elif outcome in {"cwd", "root"}:
            (process / outcome).unlink()
            (process / outcome).symlink_to(directory)
        elif outcome == "maps":
            (process / "maps").write_text(
                f"1000-2000 r--p 0000 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {info.st_ino}\n"
            )
        elif outcome == "mount":
            (process / "root").unlink()
            (process / "root").symlink_to(namespace)
            (process / "mountinfo").write_text(
                f"55 1 {os.major(info.st_dev)}:{os.minor(info.st_dev)} "
                f"{filesystem_path(directory)} /alias rw - ext4 /dev/fake rw\n"
            )

    monkeypatch.setattr(lab.usage, "sleep", resume, raising=False)
    before = lab.filesystem.snapshot(directory, hashes=True)
    usage = lab.engine.inspector.inspect((directory,), before)
    scope = usage.scope()
    if outcome in {"persistent", "replaced"}:
        assert usage.unknown and not usage.busy
        assert len(waits) == (3 if outcome == "persistent" else 1)
    else:
        assert not usage.unknown, usage.unknown
        assert len(waits) == 1
    expected_busy = outcome in {"fd", "cwd", "root", "maps", "mount"}
    assert bool(usage.busy) == expected_busy
    assert usage.exited_processes == int(outcome == "gone")
    assert scope["excluded_nondumpable_count"] == 0
    assert scope["excluded_capabilities_count"] == 0
    checks = scope["rechecks"]
    assert checks["attempts"] == len(waits)
    assert checks["max_rechecks_per_process"] == 3
    assert checks["interval_seconds"] == 0.05
    assert sum(waits) == pytest.approx(checks["wait_seconds"])
    assert checks["resolved_processes"] == int(
        outcome not in {"persistent", "replaced"}
    )
    (record,) = checks["processes"]
    assert record["pid"] == int(process.name) and record["starttime_ticks"] == 314159
    assert record["attempts"] == len(waits)
    assert record["outcome"] == (
        "unknown"
        if outcome in {"persistent", "replaced"}
        else "exited"
        if outcome == "gone"
        else "busy"
        if expected_busy
        else "inspected"
    )
    assert "maps is readable" in record["initial_error"]
    assert before.entries == lab.filesystem.snapshot(directory, hashes=True).entries


@pytest.mark.parametrize(
    "transition",
    ["last-attempt", "status-denied", "identity-denied", "credentials-changed"],
)
def test_recheck_keeps_permission_and_identity_failures_closed(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    process = private_process["path"]
    private_process["denied"] = {"fd"}
    waits: list[float] = []
    original = lab.usage.read_bytes

    def read(path: Path) -> bytes:
        if waits and (
            transition == "identity-denied"
            and path == process / "stat"
            or transition == "status-denied"
            and len(waits) < 3
            and path == process / "status"
        ):
            raise PermissionError(errno.EACCES, "transient metadata denial", str(path))
        return cast(bytes, original(path))

    def resume(delay: float) -> None:
        waits.append(delay)
        if transition == "credentials-changed":
            path = process / "status"
            path.write_bytes(path.read_bytes().replace(b"12345", b"12346"))
        if len(waits) == 3:
            private_process["denied"] = set()

    monkeypatch.setattr(lab.usage, "read_bytes", read)
    monkeypatch.setattr(lab.usage, "sleep", resume)
    observed = lab.engine.inspector.inspect(
        (directory,), lab.filesystem.snapshot(directory)
    )
    failed = transition in {"identity-denied", "credentials-changed"}
    assert bool(observed.unknown) == failed
    assert not observed.busy and observed.exited_processes == 0
    assert len(waits) == (1 if transition == "credentials-changed" else 3)
    assert observed.rechecks["resolved_processes"] == int(not failed)
    assert not observed.excluded_nondumpable and not observed.excluded_capabilities


def test_recheck_preserves_busy_observation_before_a_later_read_denial(
    lab: Lab, private_process: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    descriptor = private_process["path"] / "fd/7"
    descriptor.symlink_to(directory / "a.bin")
    private_process["denied"] = {"maps"}

    def resume(delay: float) -> None:
        descriptor.unlink()
        private_process["denied"] = set()

    monkeypatch.setattr(lab.usage, "sleep", resume)
    observed = lab.engine.inspector.inspect(
        (directory,), lab.filesystem.snapshot(directory)
    )
    assert observed.busy == ["open-fd: pid=123456789 fd=7"]
    assert not observed.unknown
    assert observed.rechecks["processes"][0]["outcome"] == "busy"


@pytest.mark.parametrize("failure", ["io-error", "missing-identity"])
def test_recheck_never_retries_io_errors_or_unpinned_identity(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    private_process["denied"] = {"fd"}
    if failure == "io-error":
        private_process["errno"] = errno.EIO
    else:
        (private_process["path"] / "stat").unlink()

    def unexpected_wait(delay: float) -> None:
        pytest.fail("cannot retry without a permission error and pinned identity")

    monkeypatch.setattr(lab.usage, "sleep", unexpected_wait)
    observed = lab.engine.inspector.inspect(
        (directory,), lab.filesystem.snapshot(directory)
    )
    assert observed.unknown and observed.exited_processes == 0
    assert observed.rechecks["attempts"] == 0


@pytest.mark.parametrize(
    "phase",
    [
        "during-resources",
        "after-resources",
        "before-recheck",
        "during-exclusion",
        "closed-proof",
    ],
)
@pytest.mark.parametrize("holds_target", [False, True], ids=["unrelated", "O_PATH"])
def test_pid_successor_is_unknown_and_preserves_observed_references(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    holds_target: bool,
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    process = private_process["path"]
    inspector = lab.engine.inspector
    private_process["denied"] = (
        set() if phase in {"during-resources", "after-resources"} else {"maps"}
    )
    descriptor = process / "fd/7"
    opened = os.open(directory / "a.bin", os.O_PATH)
    if holds_target and phase != "during-resources":
        descriptor.symlink_to(f"/proc/self/fd/{opened}")
    resumed = False
    original_resources = inspector.resources
    original_evidence = inspector.exclusion_evidence
    original_status = inspector.status

    def replace() -> None:
        path = process / "stat"
        path.write_bytes(path.read_bytes().replace(b"314159", b"314160"))

    def resources(*args: Any, **kwargs: Any) -> None:
        if phase == "during-resources":
            replace()
            if holds_target:
                descriptor.symlink_to(f"/proc/self/fd/{opened}")
        original_resources(*args, **kwargs)
        if phase == "after-resources":
            replace()

    def resume(delay: float) -> None:
        nonlocal resumed
        resumed = True
        private_process["denied"] = set()
        if phase == "before-recheck":
            replace()

    def evidence(target: Path) -> dict[str, Any]:
        if phase == "during-exclusion":
            replace()
            private_process["denied"] = {"fd", "maps", "cwd"}
        return cast(dict[str, Any], original_evidence(target))

    def status(target: Path) -> dict[str, str]:
        if phase == "closed-proof" and resumed:
            replace()
            (process / "status").write_text(
                "Name:\tfinished\nState:\tZ (zombie)\nThreads:\t1\n"
                "Uid:\t12345\t12345\t12345\t12345\n"
            )
            (process / "task" / process.name).mkdir(parents=True, exist_ok=True)
            if descriptor.is_symlink():
                descriptor.unlink()
        return cast(dict[str, str], original_status(target))

    monkeypatch.setattr(lab.usage, "sleep", resume)
    monkeypatch.setattr(inspector, "resources", resources)
    monkeypatch.setattr(inspector, "exclusion_evidence", evidence)
    monkeypatch.setattr(inspector, "status", status)
    try:
        observed = inspector.inspect((directory,), lab.filesystem.snapshot(directory))
    finally:
        os.close(opened)
    assert observed.busy == (["open-fd: pid=123456789 fd=7"] if holds_target else [])
    assert observed.unknown and "PID replaced" in observed.unknown[0]
    assert observed.exited_processes == 0
    assert not observed.excluded_nondumpable and not observed.excluded_capabilities
    if observed.rechecks["processes"]:
        assert observed.rechecks["processes"][0]["outcome"] == "unknown"
        assert observed.rechecks["resolved_processes"] == 0


@pytest.mark.parametrize("replacement", ["after-resources", "during-exclusion"])
def test_recheck_recognizes_pid_reuse_at_the_final_identity_check(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    unit = lab.unit()
    directory = lab.config.repo / unit.path
    process = private_process["path"]
    inspector = lab.engine.inspector
    private_process["denied"] = {"fd"}
    resumed = False

    def replace() -> None:
        path = process / "stat"
        path.write_bytes(path.read_bytes().replace(b"314159", b"314160"))

    def resume(delay: float) -> None:
        nonlocal resumed
        resumed = True
        private_process["denied"] = (
            set() if replacement == "after-resources" else {"fd", "maps", "cwd"}
        )

    original_resources = inspector.resources
    original_evidence = inspector.exclusion_evidence

    def resources(*args: Any, **kwargs: Any) -> None:
        original_resources(*args, **kwargs)
        if resumed and replacement == "after-resources":
            replace()

    def evidence(target: Path) -> dict[str, Any]:
        if resumed and replacement == "during-exclusion":
            replace()
        return cast(dict[str, Any], original_evidence(target))

    monkeypatch.setattr(lab.usage, "sleep", resume)
    monkeypatch.setattr(inspector, "resources", resources)
    monkeypatch.setattr(inspector, "exclusion_evidence", evidence)
    observed = inspector.inspect((directory,), lab.filesystem.snapshot(directory))
    assert observed.exited_processes == 0 and observed.unknown
    assert not observed.busy and observed.inspected_processes == 1
    assert not observed.excluded_nondumpable and not observed.excluded_capabilities
    assert observed.rechecks["processes"][0]["outcome"] == "unknown"
    assert observed.rechecks["resolved_processes"] == 0


def test_recheck_evidence_survives_later_inspections_and_is_audited(
    lab: Lab, private_process: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    private_process["denied"] = {"fd"}
    monkeypatch.setattr(
        lab.usage, "sleep", lambda delay: private_process.update(denied=set())
    )
    code, report = lab.run("mirror", unit)
    assert code == 0, report
    checks = report["process_scope"]["rechecks"]
    assert checks["attempts"] == checks["resolved_processes"] == 1
    assert checks["processes"][0]["outcome"] == "inspected"
    verified = next(row for row in lab.records() if row["event"] == "verified")
    assert verified["process_scope"]["rechecks"] == checks
    # Reusing an Engine for another command starts a new observation history.
    assert lab.run("verify", unit)[1]["process_scope"]["rechecks"]["attempts"] == 0


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


@pytest.mark.parametrize("change", ["write", "add"])
def test_exchanged_old_mirror_is_not_blindly_removed(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    local, mirror = lab.config.repo / unit.path, lab.config.mirror / unit.path
    (local / "a.bin").write_bytes(b"new output\n")
    lab.age(unit, 1)
    original = lab.module.rename
    changed = False
    old_path: Path | None = None

    def writer(source: Path, target: Path, *, exchange: bool = False) -> None:
        nonlocal changed, old_path
        directory = os.open(target, os.O_RDONLY | os.O_DIRECTORY) if exchange else -1
        try:
            original(source, target, exchange=exchange)
            if exchange and target == mirror:
                changed, old_path = True, source
                name = "a.bin" if change == "write" else "unmirrored.bin"
                fd = os.open(
                    name, os.O_WRONLY | os.O_TRUNC | os.O_CREAT, 0o600, dir_fd=directory
                )
                try:
                    os.write(fd, b"E-only generation\n")
                finally:
                    os.close(fd)
        finally:
            if directory >= 0:
                os.close(directory)

    monkeypatch.setattr(lab.module, "rename", writer)
    code, report = lab.run("mirror", unit)
    assert changed and code == 1, report
    assert old_path is not None
    name = "a.bin" if change == "write" else "unmirrored.bin"
    assert lab.filesystem.read_bytes(old_path / name) == b"E-only generation\n"
    assert lab.filesystem.read_bytes(mirror / "a.bin") == b"new output\n"
    assert not any(r["event"] == "mirror_retired" for r in lab.records())
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    result = recovered["results"][0]
    assert result["recovery_required"]
    archive = Path(result["tree_archive"])
    assert archive.is_relative_to(lab.config.mirror / ".storage-tier/recovered-trees")
    assert lab.filesystem.read_bytes(archive / name) == b"E-only generation\n"
    assert lab.run("mirror", unit)[0] == 1
    assert lab.run("verify", unit)[0] == 0
    assert not lab.engine.recovery_path(unit).exists()
    assert lab.filesystem.read_bytes(archive / name) == b"E-only generation\n"


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


@pytest.mark.parametrize("kind", ["dataset", "checkpoint-files"])
def test_hydrate_checks_e_hash_against_existing_manifest(lab: Lab, kind: str) -> None:
    if kind == "checkpoint-files":
        root = lab.config.repo / "ckpt"
        root.mkdir()
        (root / "README.md").write_bytes(b"apple\n")
        unit = lab.layout.Unit("ckpt", kind)
        filename = "README.md"
    else:
        unit = lab.unit()
        filename = "a.bin"
    lab.mirror(unit)
    if kind == "checkpoint-files":
        (root / filename).unlink()
    else:
        assert lab.run("evict", unit)[0] == 0
    manifest = lab.engine.manifest(unit)
    manifest["mirror"][filename]["sha256"] = hashlib.sha256(b"peach\n").hexdigest()
    lab.filesystem.write_json(lab.engine.manifest_path(unit), manifest)
    evidence = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    code, report = lab.run("hydrate", unit)
    assert code == 1 and "SHA-256" in report["results"][0]["error"]
    assert not (lab.config.repo / unit.path / filename).exists()
    if kind == "dataset":
        assert not (lab.config.repo / unit.path).exists()
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == evidence


@pytest.mark.parametrize("command", ["mirror", "verify", "index"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_checkpoint_root_file_operations_preserve_mismatched_hash_evidence(
    lab: Lab,
    command: str,
    dry_run: bool,
) -> None:
    root = lab.config.repo / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"apple\n")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    lab.mirror(unit)
    saved = lab.engine.manifest(unit)
    saved["mirror"]["README.md"]["sha256"] = hashlib.sha256(b"peach\n").hexdigest()
    lab.filesystem.write_json(lab.engine.manifest_path(unit), saved)
    before = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    local = lab.filesystem.snapshot(root).entries
    code, report = lab.run(command, unit, dry_run=dry_run)
    assert code == 1 and "SHA-256" in report["results"][0]["error"], report
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == before
    assert lab.filesystem.snapshot(root).entries == local


def test_checkpoint_root_file_legacy_index_preserves_mismatched_hash_evidence(
    lab: Lab,
) -> None:
    root = lab.config.mirror / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"apple\n")
    legacy = lab.layout.Unit("ckpt", "checkpoint")
    saved = lab.engine.record(
        legacy, None, lab.filesystem.snapshot(root, hashes=True), dry_run=False
    )
    saved["mirror"]["README.md"]["sha256"] = hashlib.sha256(b"peach\n").hexdigest()
    lab.filesystem.write_json(lab.engine.manifest_path(legacy), saved)
    before = lab.filesystem.read_bytes(lab.engine.manifest_path(legacy))
    code, report = lab.run("index", lab.layout.Unit("ckpt", "checkpoint-files"))
    assert code == 1 and "SHA-256" in report["results"][0]["error"], report
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(legacy)) == before
    assert not (lab.config.mirror / ".storage-tier/legacy-manifests").exists()


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


@pytest.mark.parametrize("change", ["write", "read", "add", "replace"])
def test_late_directory_fd_change_after_quarantine_snapshot_is_not_deleted(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOATIME)
    original = lab.module.snapshot
    changed = False

    def mutate_after_snapshot(path: Path, *, hashes: bool = False) -> Any:
        nonlocal changed, directory
        result = original(path, hashes=hashes)
        if path.name.startswith(".storage-tier-trash-") and not changed:
            changed = True
            if change == "replace":
                os.unlink("a.bin", dir_fd=directory)
            name = "unmirrored.bin" if change == "add" else "a.bin"
            flags = os.O_RDONLY if change == "read" else os.O_WRONLY | os.O_TRUNC
            if change in {"add", "replace"}:
                flags |= os.O_CREAT | os.O_EXCL
            child = os.open(name, flags, 0o600, dir_fd=directory)
            try:
                if change == "read":
                    assert os.read(child, 6) == b"apple\n"
                else:
                    assert os.write(child, b"pear!\n") == 6
            finally:
                os.close(child)
                os.close(directory)
                directory = -1
        return result

    monkeypatch.setattr(lab.module, "snapshot", mutate_after_snapshot)
    try:
        code, report = lab.run("evict", unit)
    finally:
        if directory >= 0:
            os.close(directory)
    assert changed and code == 1, report
    expected_name = "unmirrored.bin" if change == "add" else "a.bin"
    expected = b"apple\n" if change == "read" else b"pear!\n"
    assert lab.filesystem.read_bytes(source / expected_name) == expected
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin") == b"apple\n"
    )
    assert not any(row["event"] == "deleted" for row in lab.records())


def test_partial_removal_is_journaled_and_recovery_blocks_mirroring_until_hydrate(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    original = os.unlink
    changed = False

    def mutate_next_file(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        result = original(path, *args, **kwargs)
        fd = kwargs.get("dir_fd")
        if (
            fd is not None
            and not changed
            and Path(os.readlink(f"/proc/self/fd/{fd}")).name.startswith(
                ".storage-tier-retained-"
            )
        ):
            changed = True
            (quarantine,) = source.parent.glob(".storage-tier-trash-*")
            writer = os.open(quarantine / "nested/b.bin", os.O_WRONLY | os.O_TRUNC)
            try:
                assert os.write(writer, b"NEW!") == 4
            finally:
                os.close(writer)
        return result

    monkeypatch.setattr(os, "unlink", mutate_next_file)
    code, report = lab.run("evict", unit)
    assert changed and code == 1, report
    assert not (source / "a.bin").exists()
    assert lab.filesystem.read_bytes(source / "nested/b.bin") == b"NEW!"
    interrupted = next(r for r in lab.records() if r["event"] == "deletion_interrupted")
    assert interrupted["removed_entries"] == ["a.bin"]
    assert interrupted["partial_deletion_possible"]
    journal = [
        json.loads(line)
        for line in lab.filesystem.read_bytes(
            Path(interrupted["removal_journal"])
        ).splitlines()
    ]
    assert [r["event"] for r in journal if r["entry"] == "a.bin"] == [
        "claim_intent",
        "claimed",
        "remove_intent",
        "removed",
    ]
    assert journal[0]["expected"]["sha256"] == hashlib.sha256(b"apple\n").hexdigest()
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"][0]["recovery_required"], recovered
    archive = Path(recovered["results"][0]["recovery_plan"])
    assert (
        lab.filesystem.read_json(archive)["removal_plan"]["a.bin"]["sha256"]
        == hashlib.sha256(b"apple\n").hexdigest()
    )
    for command in ("mirror", "evict", "cycle"):
        code, blocked = lab.run(command, unit)
        assert code == 1 and "recovery" in blocked["results"][0]["error"], blocked
    # Preserve unmirrored writes; hydration must not merge over this partial tree.
    assert lab.run("hydrate", unit)[0] == 1
    preserved = tmp_path / "preserved-partial"
    source.rename(preserved)
    code, hydrated = lab.run("hydrate", unit)
    assert code == 0, hydrated
    assert lab.filesystem.read_bytes(preserved / "nested/b.bin") == b"NEW!"
    assert lab.filesystem.read_bytes(source / "a.bin") == b"apple\n"
    assert lab.filesystem.read_bytes(source / "nested/b.bin") == b"\x00\x01\x04\x09"
    assert not lab.engine.recovery_path(unit).exists()


def test_child_name_recreated_at_unlink_is_preserved_as_unverified_payload(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    original = os.unlink
    recreated = False

    def replace_at_unlink(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal recreated
        fd = kwargs.get("dir_fd")
        if fd is not None and not recreated:
            directory = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if directory.name.startswith(
                (".storage-tier-trash-", ".storage-tier-retained-")
            ):
                recreated = True
                (quarantine,) = source.parent.glob(".storage-tier-trash-*")
                (quarantine / "a.bin").write_bytes(b"unmirrored successor")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", replace_at_unlink)
    code, report = lab.run("evict", unit)
    assert recreated and code == 1, report
    assert lab.filesystem.read_bytes(source / "a.bin") == b"unmirrored successor"
    assert not any(row["event"] == "deleted" for row in lab.records())


def test_wrong_entry_claim_is_retained_when_a_successor_occupies_its_name(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    original = lab.filesystem.rename_at
    replaced = False

    def replace_at_claim(
        source_fd: int, first: str, target_fd: int, target: str
    ) -> None:
        nonlocal replaced
        if first != "a.bin" or replaced:
            original(source_fd, first, target_fd, target)
            return
        replaced = True
        os.unlink(first, dir_fd=source_fd)
        child = os.open(
            first, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=source_fd
        )
        os.write(child, b"first unmirrored generation")
        os.close(child)
        original(source_fd, first, target_fd, target)
        child = os.open(
            first, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=source_fd
        )
        os.write(child, b"second unmirrored generation")
        os.close(child)

    monkeypatch.setattr(lab.filesystem, "rename_at", replace_at_claim)
    code, report = lab.run("evict", unit)
    assert replaced and code == 1, report
    assert (
        lab.filesystem.read_bytes(source / "a.bin") == b"second unmirrored generation"
    )
    assert not any(row["event"] == "deleted" for row in lab.records())
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"][0]["recovery_required"], recovered
    assets = Path(recovered["results"][0]["retained_assets"])
    (retained_name,) = lab.filesystem.names(assets)
    assert (
        lab.filesystem.read_bytes(assets / retained_name)
        == b"first unmirrored generation"
    )
    assert (
        lab.filesystem.read_bytes(source / "a.bin") == b"second unmirrored generation"
    )
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin") == b"apple\n"
    )


def test_checkpoint_file_hardlink_does_not_make_dataset_directory_a_checkpoint(
    lab: Lab,
) -> None:
    unit, checkpoint = lab.unit(), lab.unit("ckpt/model")
    source = lab.config.repo / unit.path
    protected = lab.config.repo / checkpoint.path / "a.bin"
    (source / "a.bin").unlink()
    os.link(protected, source / "a.bin")
    lab.age(unit, 15)
    lab.mirror(unit)
    code, report = lab.run("evict", unit)
    assert code == 0 and report["results"][0]["action"] == "evicted", report
    assert lab.filesystem.read_bytes(protected) == b"apple\n"
    assert protected.stat().st_nlink == 1


def test_process_exit_after_claim_preserves_payload_and_requires_verified_recovery(
    lab: Lab,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    script = r"""
import os, sys
from pathlib import Path
import storage_tier.engine as module
from storage_tier.layout import Config
from storage_tier.usage import Inspector
original = module.append_json
def crash_after_claim(path, value):
    original(path, value)
    if value.get('event') == 'claimed':
        os._exit(91)
module.append_json = crash_after_claim
config = Config(Path(sys.argv[1]), Path(sys.argv[2]), ('data/task',))
engine = module.Engine(config, inspector=Inspector(Path(sys.argv[3])), mount_check=lambda config: None)
engine.run('evict', ['data/task/v1'], dry_run=False)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(lab.config.repo),
            str(lab.config.mount),
            str(lab.proc),
        ],
        env=dict(os.environ, PYTHONPATH=str(OPS)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 91, child.stderr
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"][0]["recovery_required"], recovered
    result = recovered["results"][0]
    assets = Path(result["retained_assets"])
    (token,) = lab.filesystem.names(assets)
    assert lab.filesystem.read_bytes(assets / token) == b"apple\n"
    source = lab.config.repo / unit.path
    assert not (source / "a.bin").exists()
    assert lab.filesystem.read_bytes(source / "nested/b.bin") == b"\x00\x01\x04\x09"
    assert lab.run("mirror", unit)[0] == 1
    assert lab.run("verify", unit)[0] == 1
    assert lab.engine.recovery_path(unit).exists()
    # Explicit repair from retained bytes, followed by full D:/E: verification.
    lab.filesystem.rename(assets / token, source / "a.bin")
    code, verified = lab.run("verify", unit)
    assert code == 0, verified
    assert not lab.engine.recovery_path(unit).exists()
    assert (source / "a.bin").exists() and (source / "nested/b.bin").exists()
    assert not any(row["event"] == "deleted" for row in lab.records())


@pytest.mark.parametrize(
    "tier,position",
    [("D", "ancestor"), ("D", "root"), ("E", "ancestor"), ("E", "root"), ("E", "repo")],
)
def test_mount_boundary_is_checked_above_and_at_unit_root(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, tier: str, position: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    base = lab.config.repo if tier == "D" else lab.config.mirror
    boundary = (
        base / ("data/task" if position == "ancestor" else unit.path)
        if position != "repo"
        else base
    )
    original = lab.filesystem.fd_mount_id

    def mounted_subtree(fd: int) -> int:
        value = cast(int, original(fd))
        current = Path(os.readlink(f"/proc/self/fd/{fd}"))
        return value + 1 if current.is_relative_to(boundary) else value

    monkeypatch.setattr(lab.filesystem, "fd_mount_id", mounted_subtree)
    try:
        code, report = lab.run("mirror" if tier == "D" else "index", unit)
    except lab.filesystem.StorageError as error:
        # Container ancestry is now checked before per-unit dispatch. The CLI
        # maps this preflight exception to failure just like a unit failure.
        assert "mount" in str(error)
    else:
        assert code == 1 and "mount" in report["results"][0]["error"], report
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin") == b"apple\n"
    )


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


def test_existing_write_fd_prevents_final_payload_removal(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    original_append, original_unlink = lab.module.append_json, os.unlink
    writer = -1
    opened = False

    def hold_writer(path: Path, value: dict[str, Any]) -> None:
        nonlocal writer, opened
        original_append(path, value)
        if value.get("event") == "claimed" and value.get("entry") == "a.bin":
            (retained,) = (lab.config.repo / unit.path).parent.glob(
                ".storage-tier-retained-*"
            )
            writer = os.open(
                retained / value["retained_name"], os.O_WRONLY | os.O_NOATIME
            )
            opened = True

    def write_after_final_check(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal writer
        if writer >= 0 and kwargs.get("dir_fd") is not None:
            os.pwrite(writer, b"pear!\n", 0)
            os.close(writer)
            writer = -1
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(lab.module, "append_json", hold_writer)
    monkeypatch.setattr(os, "unlink", write_after_final_check)
    try:
        code, report = lab.run("evict", unit)
    finally:
        if writer >= 0:
            os.close(writer)
    assert opened and code == 1, report
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin") == b"apple\n"
    )
    assert not any(r["event"] == "deleted" for r in lab.records())


@pytest.mark.parametrize("entry_path", ["token", "external-hardlink"])
def test_late_open_breaks_lease_instead_of_losing_unmirrored_payload(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry_path: str
) -> None:
    unit = lab.unit()
    source = lab.config.repo / unit.path
    alias = tmp_path / "outside.bin"
    if entry_path == "external-hardlink":
        os.link(source / "a.bin", alias)
    lab.mirror(unit)
    original = os.unlink
    attempted = False
    child_code: int | None = None
    script = r"""
import os, sys
try:
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NONBLOCK | os.O_NOATIME)
except BlockingIOError:
    sys.exit(75)
os.pwrite(fd, b'pear!\n', 0)
os.close(fd)
if sys.argv[2] == 'external-hardlink':
    os.unlink(sys.argv[1])
"""

    def open_after_final_check(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal attempted, child_code
        fd = kwargs.get("dir_fd")
        if fd is not None and not attempted:
            retained = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if retained.name.startswith(".storage-tier-retained-"):
                attempted = True
                target = alias if entry_path == "external-hardlink" else retained / path
                child = subprocess.run(
                    [sys.executable, "-c", script, str(target), entry_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                child_code = child.returncode
                assert child_code in {0, 75}, child.stderr
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", open_after_final_check)
    code, report = lab.run("evict", unit)
    assert attempted and code == 1, report
    assert child_code == 75
    if entry_path == "external-hardlink":
        assert lab.filesystem.read_bytes(alias) == b"apple\n"
    assert not any(r["event"] == "deleted" for r in lab.records())


def test_token_replacement_after_final_stat_is_not_successfully_deleted(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    removal = importlib.import_module("storage_tier.removal")
    original_hash, original_stat = removal.hashlib.file_digest, os.stat
    armed = False
    replaced = False

    def arm_after_hash(*args: Any, **kwargs: Any) -> Any:
        nonlocal armed
        result = original_hash(*args, **kwargs)
        opened = Path(os.readlink(f"/proc/self/fd/{args[0].fileno()}"))
        armed = opened.parent.name.startswith(".storage-tier-retained-")
        return result

    def replace_after_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal replaced
        result = original_stat(path, *args, **kwargs)
        fd = kwargs.get("dir_fd")
        if armed and not replaced and fd is not None:
            retained = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if retained.name.startswith(".storage-tier-retained-"):
                replaced = True
                os.unlink(path, dir_fd=fd)
                writer = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd
                )
                try:
                    os.write(writer, b"unverified token replacement\n")
                finally:
                    os.close(writer)
        return result

    monkeypatch.setattr(removal.hashlib, "file_digest", arm_after_hash)
    monkeypatch.setattr(os, "stat", replace_after_stat)
    code, report = lab.run("evict", unit)
    assert replaced and code == 1, report
    assert (
        lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin")
        == b"unverified token replacement\n"
    )
    assert not any(r["event"] == "deleted" for r in lab.records())


@pytest.mark.parametrize("failure", ["disabled", "permission", "unsupported"])
def test_unavailable_lease_never_falls_back_to_unlocked_deletion(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    original_fcntl, original_read = fcntl.fcntl, Path.read_text

    def rejected(fd: int, command: int, *args: Any) -> Any:
        if command == fcntl.F_SETLEASE and args == (fcntl.F_WRLCK,):
            code = errno.EACCES if failure == "permission" else errno.EOPNOTSUPP
            raise OSError(code, os.strerror(code))
        return original_fcntl(fd, command, *args)

    def disabled(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/proc/sys/fs/leases-enable":
            return "0\n"
        return original_read(path, *args, **kwargs)

    if failure == "disabled":
        monkeypatch.setattr(Path, "read_text", disabled)
    else:
        monkeypatch.setattr(fcntl, "fcntl", rejected)
    code, report = lab.run("evict", unit)
    assert code == 1 and "lease" in report["results"][0]["error"], report
    source = lab.config.repo / unit.path
    assert lab.filesystem.read_bytes(source / "a.bin") == b"apple\n"
    assert lab.filesystem.read_bytes(source / "nested/b.bin") == b"\x00\x01\x04\x09"
    interrupted = next(r for r in lab.records() if r["event"] == "deletion_interrupted")
    assert interrupted["removed_entries"] == []
    journal = [
        json.loads(row)
        for row in lab.filesystem.read_bytes(
            Path(interrupted["removal_journal"])
        ).splitlines()
    ]
    failed = next(r for r in journal if r["event"] == "file_removal_failed")
    assert failed["phase"] == "lease_acquire" and not failed["unlinked"]


@pytest.mark.parametrize(
    "when", ["before-unlink", "after-final-check", "after-unlink-check"]
)
def test_blocked_writer_is_released_and_lease_break_is_audited(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    unit = lab.unit()
    alias = lab.config.repo / "outside-lease-alias.bin"
    if when == "after-unlink-check":
        os.link(lab.config.repo / unit.path / "a.bin", alias)
    lab.mirror(unit)
    removal = importlib.import_module("storage_tier.removal")
    original_hash, original_unlink = removal.hashlib.file_digest, os.unlink
    original_append = lab.module.append_json
    child: subprocess.Popen[str] | None = None
    script = r"""
import os, sys
try:
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NONBLOCK | os.O_NOATIME)
except BlockingIOError:
    print('blocked', flush=True)
else:
    os.close(fd)
    sys.exit(76)
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NOATIME)
os.pwrite(fd, b'pear!\n', 0)
os.close(fd)
print('written', flush=True)
"""

    def start_writer(target: Path) -> None:
        nonlocal child
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0]
        assert child.stdout.readline().strip() == "blocked"

    def after_hash(*args: Any, **kwargs: Any) -> Any:
        result = original_hash(*args, **kwargs)
        opened = Path(os.readlink(f"/proc/self/fd/{args[0].fileno()}"))
        if (
            child is None
            and when == "before-unlink"
            and opened.parent.name.startswith(".storage-tier-retained-")
        ):
            start_writer(opened)
        return result

    def at_unlink(path: Any, *args: Any, **kwargs: Any) -> Any:
        fd = kwargs.get("dir_fd")
        if child is None and when == "after-final-check" and fd is not None:
            retained = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if retained.name.startswith(".storage-tier-retained-"):
                start_writer(retained / path)
        return original_unlink(path, *args, **kwargs)

    def after_post_unlink_check(path: Path, value: dict[str, Any]) -> None:
        original_append(path, value)
        if (
            child is None
            and when == "after-unlink-check"
            and value.get("event") == "removed"
            and value.get("entry") == "a.bin"
        ):
            start_writer(alias)

    monkeypatch.setattr(removal.hashlib, "file_digest", after_hash)
    monkeypatch.setattr(os, "unlink", at_unlink)
    monkeypatch.setattr(lab.module, "append_json", after_post_unlink_check)
    try:
        code, report = lab.run("evict", unit)
    finally:
        if child is not None:
            try:
                stdout, stderr = child.communicate(timeout=10)
                assert child.returncode == 0, stderr
                assert "written" in stdout
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
    assert child is not None and code == 1, report
    interrupted = next(r for r in lab.records() if r["event"] == "deletion_interrupted")
    journal = [
        json.loads(row)
        for row in lab.filesystem.read_bytes(
            Path(interrupted["removal_journal"])
        ).splitlines()
    ]
    failed = next(r for r in journal if r["event"] == "file_removal_failed")
    assert "lease break" in failed["error"]
    assert failed["phase"] == (
        "before_unlink" if when == "before-unlink" else "after_unlink"
    )
    assert failed["unlinked"] == (when != "before-unlink")
    assert interrupted["removal_failures"][0]["phase"] == failed["phase"]
    assert interrupted["removal_failures"][0]["unlinked"] == failed["unlinked"]
    assert interrupted["removed_entries"] == (
        [] if when == "before-unlink" else ["a.bin"]
    )
    if when == "before-unlink":
        assert (
            lab.filesystem.read_bytes(lab.config.repo / unit.path / "a.bin")
            == b"pear!\n"
        )
    assert not any(r["event"] == "deleted" for r in lab.records())


@pytest.mark.parametrize("operation", ["mirror", "hydrate"])
def test_unverified_failed_stage_is_preserved_instead_of_recursively_deleted(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    unit = lab.unit()
    if operation == "hydrate":
        lab.mirror(unit)
        assert lab.run("evict", unit)[0] == 0
    original = lab.module.copy_tree
    stage: Path | None = None

    def fail_after_new_payload(
        source: Path, target: Path, expected: Any, **kwargs: Any
    ) -> None:
        nonlocal stage
        original(source, target, expected, **kwargs)
        stage = target
        (target / "unmirrored.bin").write_bytes(b"keep failed copy payload\n")
        raise OSError(errno.EIO, "injected copy interruption")

    monkeypatch.setattr(lab.module, "copy_tree", fail_after_new_payload)
    code, report = lab.run(operation, unit)
    assert code == 1 and stage is not None, report
    assert (
        lab.filesystem.read_bytes(stage / "unmirrored.bin")
        == b"keep failed copy payload\n"
    )
    event = next(r for r in lab.records() if r["event"] == "stage_preserved")
    assert event["stage"] == str(stage)
    assert not any(r["event"] == "stage_discarded" for r in lab.records())


def test_partial_old_mirror_removal_recovers_on_e_without_replacing_new_mirror(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    mirror = lab.config.mirror / unit.path
    (source / "a.bin").write_bytes(b"new output\n")
    lab.age(unit, 1)
    original = os.unlink
    changed = False

    def write_remaining(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        result = original(path, *args, **kwargs)
        fd = kwargs.get("dir_fd")
        if not changed and fd is not None:
            directory = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if directory.name.startswith(
                ".storage-tier-retained-"
            ) and directory.is_relative_to(lab.config.mirror):
                changed = True
                (stage,) = mirror.parent.glob(".storage-tier-stage-*")
                (stage / "nested/b.bin").write_bytes(b"E-only late payload\n")
        return result

    monkeypatch.setattr(os, "unlink", write_remaining)
    code, report = lab.run("mirror", unit)
    assert changed and code == 1, report
    interrupted = next(
        r for r in lab.records() if r["event"] == "stage_removal_interrupted"
    )
    assert interrupted["operation"] == "retire-mirror"
    assert interrupted["removed_entries"] == ["a.bin"]
    assert interrupted["partial_deletion_possible"]
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    result = recovered["results"][0]
    assert result["recovery_required"]
    tree = Path(result["tree_archive"])
    assert tree.is_relative_to(lab.config.mirror)
    assert not (tree / "a.bin").exists()
    assert lab.filesystem.read_bytes(tree / "nested/b.bin") == b"E-only late payload\n"
    assert lab.filesystem.read_bytes(mirror / "a.bin") == b"new output\n"
    assert lab.run("verify", unit)[0] == 0
    assert lab.filesystem.read_bytes(tree / "nested/b.bin") == b"E-only late payload\n"


def test_process_exit_during_old_mirror_claim_archives_e_payload_on_recovery(
    lab: Lab,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    (source / "a.bin").write_bytes(b"new output\n")
    lab.age(unit, 1)
    script = r"""
import os, sys
from pathlib import Path
import storage_tier.engine as module
from storage_tier.layout import Config
from storage_tier.usage import Inspector
original = module.append_json
def crash_after_claim(path, value):
    original(path, value)
    if value.get('event') == 'claimed':
        os._exit(94)
module.append_json = crash_after_claim
config = Config(Path(sys.argv[1]), Path(sys.argv[2]), ('data/task',))
engine = module.Engine(config, inspector=Inspector(Path(sys.argv[3])), mount_check=lambda config: None)
engine.run('mirror', ['data/task/v1'], dry_run=False)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(lab.config.repo),
            str(lab.config.mount),
            str(lab.proc),
        ],
        env=dict(os.environ, PYTHONPATH=str(OPS)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert child.returncode == 94, child.stderr
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    result = recovered["results"][0]
    assets = Path(result["retained_assets"])
    assert assets.is_relative_to(lab.config.mirror / ".storage-tier/recovered-assets")
    (token,) = lab.filesystem.names(assets)
    assert lab.filesystem.read_bytes(assets / token) == b"apple\n"
    assert (
        lab.filesystem.read_bytes(lab.config.mirror / unit.path / "a.bin")
        == b"new output\n"
    )
    assert result["recovery_required"]
    assert lab.run("verify", unit)[0] == 0
    assert lab.filesystem.read_bytes(assets / token) == b"apple\n"


@pytest.mark.parametrize("operation", ["mirror", "hydrate"])
def test_verified_stage_cleanup_failure_keeps_payload_and_has_recovery(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    unit = lab.unit()
    if operation == "hydrate":
        lab.mirror(unit)
        assert lab.run("evict", unit)[0] == 0
    original_copy, original_append = lab.module.copy_tree, lab.module.append_json
    stage: Path | None = None
    changed = False

    def change_source_after_copy(
        source: Path, target: Path, expected: Any, **kwargs: Any
    ) -> None:
        nonlocal stage
        original_copy(source, target, expected, **kwargs)
        stage = target
        (source / "a.bin").write_bytes(b"source changed during copy\n")

    def change_claim(path: Path, event: dict[str, Any]) -> None:
        nonlocal changed
        original_append(path, event)
        if event.get("event") == "claimed" and event.get("entry") == "a.bin":
            assert stage is not None
            (retained,) = stage.parent.glob(".storage-tier-retained-*")
            (retained / event["retained_name"]).write_bytes(
                b"stage-only late payload\n"
            )
            changed = True

    monkeypatch.setattr(lab.module, "copy_tree", change_source_after_copy)
    monkeypatch.setattr(lab.module, "append_json", change_claim)
    code, report = lab.run(operation, unit)
    assert changed and code == 1, report
    assert stage is not None
    assert lab.filesystem.read_bytes(stage / "a.bin") == b"stage-only late payload\n"
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    result = recovered["results"][0]
    tree = Path(result["tree_archive"])
    base = lab.config.mirror if operation == "mirror" else lab.config.repo
    assert tree.is_relative_to(base / ".storage-tier/recovered-trees")
    assert lab.filesystem.read_bytes(tree / "a.bin") == b"stage-only late payload\n"
    assert not result["recovery_required"]


@pytest.mark.parametrize(
    "operation,timing",
    [
        ("evict", "after-unlink-check"),
        ("evict", "during-unlock"),
        ("mirror", "after-unlink-check"),
        ("mirror", "during-unlock"),
    ],
)
def test_late_nonblocking_break_cannot_complete_a_removal_transaction(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    timing: str,
) -> None:
    unit = lab.unit()
    alias = tmp_path / "external-lease-alias.bin"
    original_copy = lab.module.copy_tree
    if operation == "evict":
        os.link(lab.config.repo / unit.path / "a.bin", alias)
    else:

        def link_old_mirror(
            source: Path, target: Path, expected: Any, **kwargs: Any
        ) -> None:
            original_copy(source, target, expected, **kwargs)
            if not alias.exists():
                os.link(target / "a.bin", alias)

        monkeypatch.setattr(lab.module, "copy_tree", link_old_mirror)
    lab.mirror(unit)
    if operation == "mirror":
        (lab.config.repo / unit.path / "a.bin").write_bytes(b"new output\n")
        lab.age(unit, 1)
    original_append, original_fcntl = lab.module.append_json, fcntl.fcntl
    broken = False
    script = r"""
import os, sys
try:
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NONBLOCK | os.O_NOATIME)
except BlockingIOError:
    sys.exit(75)
os.close(fd)
sys.exit(76)
"""

    def request_break() -> None:
        nonlocal broken
        broken = True
        child = subprocess.run(
            [sys.executable, "-c", script, str(alias)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert child.returncode == 75, child.stderr

    def after_check(path: Path, value: dict[str, Any]) -> None:
        original_append(path, value)
        if (
            not broken
            and timing == "after-unlink-check"
            and value.get("event") == "removed"
            and value.get("entry") == "a.bin"
        ):
            request_break()

    def during_unlock(fd: int, command: int, *args: Any) -> Any:
        if (
            not broken
            and timing == "during-unlock"
            and command == fcntl.F_SETLEASE
            and args == (fcntl.F_UNLCK,)
        ):
            info, external = os.fstat(fd), alias.stat()
            if (info.st_dev, info.st_ino) == (external.st_dev, external.st_ino):
                request_break()
        return original_fcntl(fd, command, *args)

    monkeypatch.setattr(lab.module, "append_json", after_check)
    monkeypatch.setattr(fcntl, "fcntl", during_unlock)
    code, report = lab.run(operation, unit)
    assert broken and code == 1, report
    event = (
        "deletion_interrupted" if operation == "evict" else "stage_removal_interrupted"
    )
    audit = next(r for r in lab.records() if r["event"] == event)
    failed = audit["removal_failures"][0]
    assert failed["phase"] == "after_unlink" and failed["unlinked"]
    assert audit["removed_entries"] == ["a.bin"]
    assert lab.filesystem.read_bytes(alias) == b"apple\n"
    assert not any(r["event"] in {"deleted", "mirror_retired"} for r in lab.records())
    assert lab.filesystem.names(lab.config.state / "transactions")


def test_opath_reference_is_busy_and_can_reopen_an_unlinked_inode(lab: Lab) -> None:
    unit = lab.unit()
    source = lab.config.repo / unit.path
    payload = source / "a.bin"
    tree = lab.filesystem.snapshot(source)
    process = lab.proc / str(os.getpid())
    process.symlink_to(f"/proc/{os.getpid()}", target_is_directory=True)
    before_reference = lab.usage.Inspector(lab.proc).inspect((source,), tree)
    opath = os.open(payload, os.O_PATH)
    lease_fd = -1
    reopened = -1
    try:
        usage = lab.usage.Inspector(lab.proc).inspect((source,), tree)
        matched = any(f"fd={opath}" == row.split()[-1] for row in usage.busy)
        initial_nlink = os.fstat(opath).st_nlink
        lease_fd = os.open(payload, os.O_RDONLY | os.O_NOATIME)
        leases = importlib.import_module("storage_tier.leases")
        with leases.write_lease(lease_fd):
            before = fcntl.fcntl(lease_fd, fcntl.F_GETLEASE)
            os.unlink(payload)
            after = fcntl.fcntl(lease_fd, fcntl.F_GETLEASE)
        reopened = os.open(f"/proc/self/fd/{opath}", os.O_RDWR | os.O_NOATIME)
        os.pwrite(reopened, b"pear!\n", 0)
        observed = {
            "kernel": os.uname().release,
            "reference": "O_PATH",
            "before_reference_busy": bool(before_reference.busy),
            "initial_nlink": initial_nlink,
            "existing_reference_busy": matched,
            "unknown": usage.unknown,
            "lease_before_unlink": before,
            "lease_after_unlink": after,
            "reopened_nlink": os.fstat(reopened).st_nlink,
            "reopened_payload": os.pread(reopened, 6, 0).decode(),
            "source_exists": payload.exists(),
        }
        print(json.dumps(observed, sort_keys=True))
        assert not before_reference.busy and not before_reference.unknown, observed
        assert matched and not usage.unknown, observed
        assert initial_nlink == 1 and before == after == fcntl.F_WRLCK, observed
        assert (
            observed["reopened_nlink"] == 0
            and observed["reopened_payload"] == "pear!\n"
        ), observed
        assert not payload.exists()
    finally:
        for fd in (reopened, lease_fd, opath):
            if fd >= 0:
                os.close(fd)


@pytest.mark.parametrize(
    "condition",
    ["match", "other-inode", "other-device", "malformed", "denied", "missing"],
)
def test_mount_reference_inspection_uses_identity_and_fails_closed(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    condition: str,
) -> None:
    unit = lab.unit()
    private_process["denied"] = set()
    process = private_process["path"]
    namespace_root = tmp_path / "namespace-root"
    namespace_root.mkdir()
    alias = namespace_root / "outside alias\\tail.bin"
    if condition == "match":
        os.link(lab.config.repo / unit.path / "a.bin", alias)
    else:
        alias.write_bytes(b"unrelated namespace file\n")
    (process / "root").unlink()
    (process / "root").symlink_to(namespace_root, target_is_directory=True)
    lab.mirror(unit)
    info = (lab.config.repo / unit.path / "a.bin").stat()
    device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
    if condition == "other-device":
        device = "0:99999" if info.st_dev != os.makedev(0, 99999) else "0:99998"
        alias.unlink()  # Unrelated devices must not trigger a namespace path stat.
    if condition == "missing":
        alias.unlink()
    text = f"55 1 {device} {filesystem_path(lab.config.repo / unit.path)} /outside\\040alias\\134tail.bin rw - ext4 /dev/fake rw\n"
    if condition == "malformed":
        text = "55 incomplete mount record\n"
    (process / "mountinfo").write_text(text)
    if condition == "denied":
        original = lab.usage.read_bytes

        def denied(path: Path) -> bytes:
            if path == process / "mountinfo":
                raise PermissionError(
                    errno.EACCES, "injected mountinfo denial", str(path)
                )
            return cast(bytes, original(path))

        monkeypatch.setattr(lab.usage, "read_bytes", denied)
    code, report = lab.run("evict", unit)
    result = report["results"][0]
    if condition == "match":
        assert (
            code == 0
            and result["action"] == "kept"
            and "mount-reference" in result["reason"]
        ), report
        assert lab.filesystem.read_bytes(alias) == b"apple\n"
    elif condition == "other-device":
        assert code == 0 and result["action"] == "evicted", report
    else:
        assert code == 1 and "usage cannot be verified" in result["error"], report
    assert (lab.config.repo / unit.path).exists() == (condition != "other-device")


@pytest.mark.parametrize("operation", ["evict", "retire-mirror"])
def test_stacked_mount_with_hidden_unit_inode_blocks_removal(
    lab: Lab,
    private_process: dict[str, Any],
    tmp_path: Path,
    operation: str,
) -> None:
    private_process["denied"] = set()
    unit = lab.unit()
    lab.mirror(unit)
    source = (
        lab.config.repo if operation == "evict" else lab.config.mirror
    ) / unit.path
    original_inode = (source / "a.bin").stat().st_ino
    namespace = tmp_path / "covered-mount"
    namespace.mkdir()
    cover = namespace / "overlaid.bin"
    cover.write_bytes(b"covering inode")
    process = private_process["path"]
    (process / "root").unlink()
    (process / "root").symlink_to(namespace, target_is_directory=True)
    device = source.stat().st_dev
    prefix = f"{os.major(device)}:{os.minor(device)}"
    lower = filesystem_path(source / "a.bin")
    upper = filesystem_path(cover)
    (process / "mountinfo").write_text(
        f"55 1 {prefix} {lower} /overlaid.bin rw - ext4 /dev/fake rw\n"
        f"56 1 {prefix} {upper} /overlaid.bin rw - ext4 /dev/fake rw\n"
    )
    if operation == "retire-mirror":
        (lab.config.repo / unit.path / "a.bin").write_bytes(b"peach\n")
    code, report = lab.run("evict" if operation == "evict" else "mirror", unit)
    assert code == 1 and "usage cannot be verified" in report["results"][0]["error"], (
        report
    )
    assert (source / "a.bin").stat().st_ino == original_inode
    assert lab.filesystem.read_bytes(source / "a.bin") == b"apple\n"


def test_inaccessible_ancestor_bind_does_not_block_verifying_e(
    lab: Lab, private_process: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    private_process["denied"] = set()
    unit = lab.unit()
    lab.mirror(unit)
    process = private_process["path"]
    mirror = lab.config.mirror / unit.path
    info = mirror.stat()
    root = filesystem_path(mirror.parent)
    (process / "mountinfo").write_text(
        f"288 83 {os.major(info.st_dev)}:{os.minor(info.st_dev)} {root} /var/lib/tennis-lab-actions/assets/data ro,noatime - ext4 /dev/fake rw\n"
    )
    original = Path.stat
    denied = 0
    target = process / "root/var/lib/tennis-lab-actions/assets/data"

    def inaccessible(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal denied
        if path == target:
            denied += 1
            raise PermissionError(
                errno.EACCES, "CI assets parent is inaccessible", str(path)
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible)
    code, report = lab.run("verify", unit, dry_run=True)
    assert code == 0 and report["results"][0]["action"] == "would_verify", report
    assert denied == 0

    scope = report["process_scope"]["mount_scope"]
    assert scope["inspection_enabled"] is False and scope["checked_roots"] == 0
    code, verified = lab.run("verify", unit)
    assert code == 0, verified
    audit = [row for row in lab.records() if row["event"] == "verified"][-1]
    assert audit["process_scope"]["mount_scope"]["inspection_enabled"] is False
    assert denied == 0


def test_checkpoint_children_and_root_files_preserve_bound_root(lab: Lab) -> None:
    root = lab.config.repo / "ckpt"
    (root / "model").mkdir(parents=True)
    (root / "model/weights").write_bytes(b"weights")
    (root / "README.md").write_bytes(b"first")
    inventory = lab.layout.discover(lab.config, lab.config.repo)
    assert {(u.path, u.kind) for u in inventory.units} == {
        ("ckpt/model", "checkpoint"),
        ("ckpt", "checkpoint-files"),
    }
    assert "ckpt/README.md" not in inventory.unmanaged
    code, report = lab.engine.run("mirror", [], dry_run=False)
    assert code == 0, report
    mirror = lab.config.mirror / "ckpt"
    root_inode, model_inode = mirror.stat().st_ino, (mirror / "model").stat().st_ino
    (root / "README.md").write_bytes(b"second")
    code, report = lab.engine.run("mirror", ["ckpt/README.md"], dry_run=False)
    assert code == 0, report
    assert mirror.stat().st_ino == root_inode
    assert (mirror / "model").stat().st_ino == model_inode
    assert lab.filesystem.read_bytes(mirror / "README.md") == b"second"
    history = lab.config.mirror / ".storage-tier/checkpoint-file-history"
    assert any(
        lab.filesystem.read_bytes(p) == b"first" for p in history.rglob("README.md")
    )
    code, report = lab.engine.run("evict", [], dry_run=False)
    assert code == 0 and all(r["action"] == "kept" for r in report["results"]), report
    local_root_inode = root.stat().st_ino
    (root / "README.md").unlink()
    code, report = lab.engine.run("hydrate", ["ckpt"], dry_run=False)
    assert code == 0 and lab.filesystem.read_bytes(root / "README.md") == b"second", (
        report
    )
    assert root.stat().st_ino == local_root_inode and (root / "model/weights").exists()


def test_root_file_removal_archives_payload_and_dry_run_is_immutable(lab: Lab) -> None:
    root = lab.config.repo / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"retain this")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    lab.mirror(unit)
    mirror = lab.config.mirror / "ckpt"
    before = lab.filesystem.snapshot(mirror, hashes=True).entries
    root_inode = mirror.stat().st_ino
    (root / "README.md").unlink()
    assert lab.run("mirror", unit, dry_run=True)[0] == 0
    assert before == lab.filesystem.snapshot(mirror, hashes=True).entries
    assert lab.run("mirror", unit)[0] == 0
    assert not (mirror / "README.md").exists() and mirror.stat().st_ino == root_inode
    history = list(
        (lab.config.mirror / ".storage-tier/checkpoint-file-history").glob(
            "*/removed-from-source/README.md"
        )
    )
    assert len(history) == 1 and lab.filesystem.read_bytes(history[0]) == b"retain this"


def test_explicit_root_file_index_archives_legacy_manifest(lab: Lab) -> None:
    checkpoint = lab.unit("ckpt/model")
    lab.mirror(checkpoint)
    legacy = lab.layout.Unit("ckpt", "checkpoint")
    old = lab.engine.record(
        legacy,
        None,
        lab.filesystem.snapshot(lab.config.mirror / "ckpt", hashes=True),
        dry_run=False,
    )
    root_files = lab.layout.Unit("ckpt", "checkpoint-files")
    assert lab.run("mirror", root_files)[0] == 1
    assert lab.run("index", root_files)[0] == 0
    assert lab.engine.manifest(root_files)["kind"] == "checkpoint-files"
    archived = list(
        (lab.config.mirror / ".storage-tier/legacy-manifests").glob("*.json")
    )
    assert len(archived) == 1 and lab.filesystem.read_json(archived[0]) == old
    assert lab.run("verify", checkpoint)[0] == 0


def test_legacy_whole_checkpoint_cannot_be_published(lab: Lab) -> None:
    lab.unit("ckpt/model")
    with pytest.raises(lab.filesystem.StorageError, match="whole-ckpt"):
        lab.engine.mirror(lab.layout.Unit("ckpt", "checkpoint"), dry_run=False)


def test_checkpoint_root_files_preserve_independent_container_permissions(
    lab: Lab,
) -> None:
    local, mirror = lab.config.repo / "ckpt", lab.config.mirror / "ckpt"
    local.mkdir(mode=0o700)
    mirror.mkdir(mode=0o755)
    mirror.chmod(0o755)
    (local / "README.md").write_bytes(b"root metadata is not mirrored payload")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    local_inode, mirror_inode = local.stat().st_ino, mirror.stat().st_ino
    code, report = lab.run("mirror", unit)
    assert code == 0, report
    assert lab.run("verify", unit)[0] == 0
    assert (local.stat().st_ino, mirror.stat().st_ino) == (local_inode, mirror_inode)
    assert local.stat().st_mode & 0o777 == 0o700
    assert mirror.stat().st_mode & 0o777 == 0o755
    (local / "README.md").unlink()
    assert lab.run("hydrate", unit)[0] == 0
    assert (
        lab.filesystem.read_bytes(local / "README.md")
        == b"root metadata is not mirrored payload"
    )
    assert (local.stat().st_ino, mirror.stat().st_ino) == (local_inode, mirror_inode)
    assert local.stat().st_mode & 0o777 == 0o700
    assert mirror.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("tamper", [None, "source", "destination"])
@pytest.mark.parametrize("operation", ["mirror", "hydrate"])
def test_interrupted_root_file_publication_keeps_old_inode_and_blocks_retry(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str | None,
    operation: str,
) -> None:
    root = lab.config.repo / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"first")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    lab.mirror(unit)
    old_manifest = lab.engine.manifest(unit)
    if operation == "mirror":
        (root / "README.md").write_bytes(b"second")
    else:
        (root / "README.md").unlink()
    expected_payload = b"second" if operation == "mirror" else b"first"
    module = importlib.import_module("storage_tier.checkpoint_files")
    original = module.rename

    def interrupted(first: Path, second: Path, **kwargs: Any) -> None:
        original(first, second, **kwargs)
        raise OSError(errno.EIO, "after exchange")

    monkeypatch.setattr(module, "rename", interrupted)
    code, report = lab.run(operation, unit)
    assert code == 1, report
    destination = lab.config.mirror if operation == "mirror" else lab.config.repo
    assert lab.filesystem.read_bytes(destination / "ckpt/README.md") == expected_payload
    if operation == "mirror":
        assert any(
            lab.filesystem.read_bytes(p) == b"first"
            for p in (
                lab.config.mirror / ".storage-tier/checkpoint-file-history"
            ).rglob("README.md")
        )
    assert lab.records()[-2]["event"] == "checkpoint_files_interrupted"
    monkeypatch.setattr(module, "rename", original)
    code, report = lab.run("mirror", unit)
    assert (
        code == 1
        and "unfinished checkpoint-file publication" in report["results"][0]["error"]
    )
    if tamper:
        base = (
            (lab.config.repo if operation == "mirror" else lab.config.mirror)
            if tamper == "source"
            else destination
        )
        (base / "ckpt/README.md").write_bytes(b"unverified")
        before = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
        with pytest.raises(lab.filesystem.StorageError, match="SHA-256|publication"):
            lab.engine.run("recover", [], dry_run=False)
        assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == before
        assert lab.run("mirror", unit)[0] == 1
        return
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"], recovered
    item = next(
        r for r in recovered["results"] if r["action"] == "publication_recovered"
    )
    evidence = lab.filesystem.read_json(Path(item["evidence_archive"]))
    assert evidence["previous_manifest"] == old_manifest
    assert lab.run("verify", unit)[0] == 0
    assert lab.run("mirror", unit)[0] == 0
    (root / "README.md").unlink()
    assert lab.run("hydrate", unit)[0] == 0
    assert lab.filesystem.read_bytes(root / "README.md") == expected_payload


@pytest.mark.parametrize("operation", ["mirror", "hydrate"])
@pytest.mark.parametrize("boundary", ["before-rename", "before-record"])
def test_directory_publication_interruption_recovers_previous_evidence(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    boundary: str,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    if operation == "mirror":
        (lab.config.repo / unit.path / "a.bin").write_bytes(b"peach\n")
    else:
        assert lab.run("evict", unit)[0] == 0
    prior = lab.engine.manifest(unit)
    target = lab.module if boundary == "before-rename" else lab.engine
    method = "rename" if boundary == "before-rename" else "record"
    original = getattr(target, method)

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise OSError(errno.EIO, boundary)

    monkeypatch.setattr(target, method, interrupted)
    assert lab.run(operation, unit)[0] == 1
    monkeypatch.setattr(target, method, original)
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"], recovered
    item = next(
        r for r in recovered["results"] if r["action"] == "publication_recovered"
    )
    assert (
        lab.filesystem.read_json(Path(item["evidence_archive"]))["previous_manifest"]
        == prior
    )
    if operation == "mirror":
        assert (
            lab.filesystem.read_bytes(Path(item["old_tree_archive"]) / "a.bin")
            == b"apple\n"
        )
    else:
        assert item["old_tree_archive"] is None
    assert lab.run("verify", unit)[0] == 0
    assert lab.run("mirror", unit)[0] == 0
    assert lab.run("evict", unit)[0] == 0
    # Hydration gives a fresh atime; the first eviction may therefore keep it.
    if (lab.config.repo / unit.path).exists():
        lab.age(unit, 15)
        assert lab.run("evict", unit)[0] == 0
    assert lab.run("hydrate", unit)[0] == 0


@pytest.mark.parametrize("command", ["index", "verify", "mirror", "hydrate"])
def test_read_only_mount_reference_is_checked_only_on_removal(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    private_process["denied"] = set()
    unit = lab.unit()
    lab.mirror(unit)
    mirror = lab.config.mirror / unit.path
    info = mirror.stat()
    process = private_process["path"]
    root = filesystem_path(mirror)
    (process / "mountinfo").write_text(
        f"288 83 {os.major(info.st_dev)}:{os.minor(info.st_dev)} {root} /hidden ro - ext4 /dev/fake rw\n"
    )
    original = Path.stat

    def inaccessible(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == process / "root/hidden":
            raise PermissionError(errno.EACCES, "inaccessible root bind", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible)
    if command == "hydrate":
        lab.filesystem.rename(lab.config.repo / unit.path, lab.config.repo / "saved")
    code, report = lab.run(command, unit, dry_run=command != "index")
    assert code == 0, report
    inspected = lab.engine.inspector.inspect((mirror,), lab.filesystem.snapshot(mirror))
    # Each bounded recheck repeats the mount-reference inspection too.
    assert inspected.unknown and inspected.mount_scope["checked_roots"] == 4


@pytest.mark.parametrize(
    "relation",
    ["unit", "file", "ancestor", "sibling", "prefix", "deleted", "inside-denied"],
)
def test_mount_root_scope_has_component_boundaries_and_never_stats_excluded_roots(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relation: str,
) -> None:
    private_process["denied"] = set()
    unit = lab.unit("data/task/a")
    path = lab.config.repo / unit.path
    process = private_process["path"]
    namespace = tmp_path / "consumer-root"
    namespace.mkdir()
    point = namespace / "restricted/alias"
    point.parent.mkdir()
    point.symlink_to(
        path if relation == "unit" else path / "a.bin",
        target_is_directory=relation == "unit",
    )
    (process / "root").unlink()
    (process / "root").symlink_to(namespace, target_is_directory=True)
    root = Path(filesystem_path(path))
    raw = str(root)
    if relation == "file" or relation == "inside-denied":
        raw = str(root / "a.bin")
    elif relation == "ancestor":
        raw = str(root.parent)
    elif relation == "sibling":
        raw = str(root.with_name("sibling"))
    elif relation == "prefix":
        raw = str(root) + "b"
    elif relation == "deleted":
        raw = str(root) + "//deleted"
    info = path.stat()
    (process / "mountinfo").write_text(
        f"288 83 {os.major(info.st_dev)}:{os.minor(info.st_dev)} {raw} /restricted/alias ro,noatime - ext4 /dev/fake rw\n"
    )
    original = Path.stat
    stat_calls = 0

    def restricted(target: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal stat_calls
        if target == process / "root/restricted/alias":
            stat_calls += 1
            if relation not in {"unit", "file"}:
                raise PermissionError(
                    errno.EACCES, "inaccessible bind parent", str(target)
                )
        return original(target, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", restricted)
    result = lab.engine.inspector.inspect((path,), lab.filesystem.snapshot(path))
    scope = result.scope()["mount_scope"]
    if relation in {"unit", "file"}:
        assert result.busy and not result.unknown
        assert scope["checked_roots"] == 1 and stat_calls == 1
    elif relation in {"deleted", "inside-denied"}:
        assert result.unknown
        assert stat_calls == (4 if relation == "inside-denied" else 0)
    else:
        assert not result.busy and not result.unknown
        assert stat_calls == 0 and scope["checked_roots"] == 0
        assert (
            scope[
                "excluded_ancestor_roots"
                if relation == "ancestor"
                else "excluded_outside_roots"
            ]
            == 1
        )


@pytest.mark.parametrize(
    "name",
    [
        "v1",
        ".storage-tier-trash-id",
        ".storage-tier-stage-id",
        ".storage-tier-retained-id",
        r"v\040",
        r"v\q",
    ],
)
def test_filesystem_unit_uses_caller_mount_root_for_original_and_temporary_names(
    lab: Lab,
    private_process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    private_process["denied"] = set()
    unit = lab.unit("data/task/" + name)
    path = lab.config.repo / unit.path
    process = private_process["path"]
    namespace = tmp_path / "consumer-root"
    namespace.mkdir()
    (namespace / "alias").symlink_to(path / "a.bin")
    (process / "root").unlink()
    (process / "root").symlink_to(namespace, target_is_directory=True)
    info = path.stat()
    record = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(path), "--output", "ID"], text=True
        )
    )["filesystems"][0]
    device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
    own = f"{record['id']} 1 {device} /volume {lab.config.repo} rw - ext4 /dev/fake rw\n".encode()
    expected = Path("/volume") / unit.path
    escaped = str(expected).replace("\\", "\\134").replace(" ", "\\040")
    (process / "mountinfo").write_text(
        f"80 1 {device} /consumer-only / rw - ext4 /dev/fake rw\n81 80 {device} {escaped}/a.bin /alias rw - ext4 /dev/fake rw\n"
    )
    original = lab.engine.inspector.metadata

    def metadata(target: Path, entry: str) -> bytes:
        if target == Path("/proc/self") and entry == "mountinfo":
            return own
        return cast(bytes, original(target, entry))

    monkeypatch.setattr(lab.engine.inspector, "metadata", metadata)
    result = lab.engine.inspector.inspect((path,), lab.filesystem.snapshot(path))
    assert result.busy and not result.unknown, result
    assert result.scope()["mount_scope"]["filesystem_unit"] == str(expected)
    assert result.scope()["mount_scope"]["checked_roots"] == 1


@pytest.mark.parametrize("failure", ["missing-mount", "deleted-root", "wrong-point"])
def test_unverifiable_caller_filesystem_mapping_never_uses_consumer_namespace(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    unit = lab.unit()
    path = lab.config.repo / unit.path
    info = path.stat()
    record = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(path), "--output", "ID"], text=True
        )
    )["filesystems"][0]
    mount_id = record["id"] + (1 if failure == "missing-mount" else 0)
    root = "//deleted/source" if failure == "deleted-root" else "/"
    point = "/not-the-unit" if failure == "wrong-point" else "/"
    raw = f"{mount_id} 1 {os.major(info.st_dev)}:{os.minor(info.st_dev)} {root} {point} rw - ext4 /dev/fake rw\n".encode()
    original = lab.engine.inspector.metadata

    def metadata(target: Path, entry: str) -> bytes:
        if target == Path("/proc/self") and entry == "mountinfo":
            return raw
        return cast(bytes, original(target, entry))

    monkeypatch.setattr(lab.engine.inspector, "metadata", metadata)
    result = lab.engine.inspector.inspect((path,), lab.filesystem.snapshot(path))
    assert result.unknown and "unit filesystem path" in result.unknown[0]


def test_checkpoint_recovery_completes_partial_verified_file_set(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = lab.config.repo / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"old readme")
    (root / "notes.txt").write_bytes(b"old notes")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    lab.mirror(unit)
    prior = lab.engine.manifest(unit)
    (root / "README.md").write_bytes(b"new readme")
    (root / "notes.txt").write_bytes(b"new notes")
    module = importlib.import_module("storage_tier.checkpoint_files")
    original = module.rename

    def interrupt(first: Path, second: Path, **kwargs: Any) -> None:
        original(first, second, **kwargs)
        raise OSError(errno.EIO, "after first publication")

    monkeypatch.setattr(module, "rename", interrupt)
    assert lab.run("mirror", unit)[0] == 1
    mirror = lab.config.mirror / "ckpt"
    assert lab.filesystem.read_bytes(mirror / "README.md") == b"new readme"
    assert lab.filesystem.read_bytes(mirror / "notes.txt") == b"old notes"
    monkeypatch.setattr(module, "rename", original)
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and report["results"], report
    assert lab.run("verify", unit)[0] == 0
    assert lab.filesystem.read_bytes(mirror / "notes.txt") == b"new notes"
    archive = Path(report["results"][0]["evidence_archive"])
    assert lab.filesystem.read_json(archive)["previous_manifest"] == prior
    assert lab.engine.run("recover", [], dry_run=False)[1]["results"] == []


@pytest.mark.parametrize("tamper", ["source", "destination"])
def test_directory_publication_recovery_rejects_unverified_payload(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    (lab.config.repo / unit.path / "a.bin").write_bytes(b"peach\n")
    original = lab.engine.record

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise OSError(errno.EIO, "before manifest")

    monkeypatch.setattr(lab.engine, "record", interrupt)
    assert lab.run("mirror", unit)[0] == 1
    monkeypatch.setattr(lab.engine, "record", original)
    base = lab.config.repo if tamper == "source" else lab.config.mirror
    (base / unit.path / "a.bin").write_bytes(b"unverified")
    manifest = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    with pytest.raises(lab.filesystem.StorageError, match="SHA-256|publication"):
        lab.engine.run("recover", [], dry_run=False)
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == manifest


@pytest.mark.parametrize("kind", ["directory", "checkpoint-files"])
def test_publication_recovery_can_retry_after_evidence_commit_failure(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "checkpoint-files":
        root = lab.config.repo / "ckpt"
        root.mkdir()
        payload = root / "README.md"
        payload.write_bytes(b"old")
        unit = lab.layout.Unit("ckpt", kind)
    else:
        unit = lab.unit()
        payload = lab.config.repo / unit.path / "a.bin"
    lab.mirror(unit)
    prior = lab.engine.manifest(unit)
    payload.write_bytes(b"new")
    original = lab.engine.record

    def failed_commit(*args: Any, **kwargs: Any) -> Any:
        raise OSError(errno.EIO, "manifest commit failed")

    monkeypatch.setattr(lab.engine, "record", failed_commit)
    assert lab.run("mirror", unit)[0] == 1
    with pytest.raises(OSError, match="manifest commit failed"):
        lab.engine.run("recover", [], dry_run=False)
    archive = lab.config.state / "publication-evidence"
    evidence_before = {p: lab.filesystem.read_bytes(p) for p in archive.iterdir()}
    assert len(evidence_before) == 1
    assert lab.engine.manifest(unit) == prior
    monkeypatch.setattr(lab.engine, "record", original)
    code, result = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and result["results"], result
    assert evidence_before == {
        p: lab.filesystem.read_bytes(p) for p in archive.iterdir()
    }
    assert lab.run("verify", unit)[0] == 0
    assert lab.engine.run("recover", [], dry_run=False)[1]["results"] == []


def test_unverified_checkpoint_copy_can_only_abort_without_reissuing_evidence(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = lab.config.repo / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"old")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    lab.mirror(unit)
    prior = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    (root / "README.md").write_bytes(b"new")
    module = importlib.import_module("storage_tier.checkpoint_files")
    original = module.copy_tree

    def failed_copy(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EIO, "copy is unverified")

    monkeypatch.setattr(module, "copy_tree", failed_copy)
    assert lab.run("mirror", unit)[0] == 1
    monkeypatch.setattr(module, "copy_tree", original)
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and recovered["results"][0]["action"] == "publication_aborted"
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == prior
    assert lab.filesystem.read_bytes(lab.config.mirror / "ckpt/README.md") == b"old"
    assert lab.run("mirror", unit)[0] == 0


@pytest.mark.parametrize("kind", ["directory", "checkpoint-files"])
def test_recovery_requires_durable_intent_even_when_both_tiers_match(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "checkpoint-files":
        root = lab.config.repo / "ckpt"
        root.mkdir()
        local = root / "README.md"
        local.write_bytes(b"old")
        unit = lab.layout.Unit("ckpt", kind)
    else:
        unit = lab.unit()
        local = lab.config.repo / unit.path / "a.bin"
    lab.mirror(unit)
    before = lab.filesystem.read_bytes(lab.engine.manifest_path(unit))
    local.write_bytes(b"planned")
    original = lab.engine.record

    def stopped(*args: Any, **kwargs: Any) -> Any:
        raise OSError(errno.EIO, "before recording published evidence")

    monkeypatch.setattr(lab.engine, "record", stopped)
    assert lab.run("mirror", unit)[0] == 1
    monkeypatch.setattr(lab.engine, "record", original)
    mirror = lab.config.mirror / local.relative_to(lab.config.repo)
    local.write_bytes(b"same unverified replacement")
    mirror.write_bytes(b"same unverified replacement")
    with pytest.raises(lab.filesystem.StorageError, match="SHA-256"):
        lab.engine.run("recover", [], dry_run=False)
    assert lab.filesystem.read_bytes(lab.engine.manifest_path(unit)) == before
    assert local.exists() and mirror.exists()


def test_completed_publication_receipt_recovers_old_stage_and_daily_cycle(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    source = lab.config.repo / unit.path
    (source / "a.bin").write_bytes(b"new verified output\n")
    lab.age(unit, 1)
    module = importlib.import_module("storage_tier.publication")
    original = module.complete

    class Stopped(BaseException):
        pass

    def crash_after_complete(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise Stopped

    with monkeypatch.context() as patch:
        patch.setattr(module, "complete", crash_after_complete)
        with pytest.raises(Stopped):
            lab.run("mirror", unit)
    assert list((lab.config.mirror / unit.path).parent.glob(".storage-tier-stage-*"))
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    assert lab.run("verify", unit)[0] == 0
    code, report = lab.engine.run("cycle", [], dry_run=True)
    assert code == 0, report
    assert not report["layout_errors"]


@pytest.mark.parametrize("archive_changed", [False, True])
def test_legacy_checkpoint_index_can_retry_after_archiving_old_manifest(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, archive_changed: bool
) -> None:
    root = lab.config.mirror / "ckpt"
    root.mkdir()
    (root / "README.md").write_bytes(b"legacy checkpoint documentation")
    unit = lab.layout.Unit("ckpt", "checkpoint-files")
    legacy = lab.layout.Unit("ckpt", "checkpoint")
    tree = lab.filesystem.snapshot(root, hashes=True)
    lab.engine.record(legacy, None, tree, dry_run=False)
    module = importlib.import_module("storage_tier.checkpoint_files")
    original = module.write_json

    def interrupted(path: Path, value: dict[str, Any]) -> None:
        original(path, value)
        if path.parent.name == "legacy-manifests":
            raise OSError("after archiving the previous manifest")

    with monkeypatch.context() as patch:
        patch.setattr(module, "write_json", interrupted)
        assert lab.run("index", unit)[0] == 1
    archives = lab.config.mirror / ".storage-tier/legacy-manifests"
    before = {p: lab.filesystem.read_bytes(p) for p in archives.iterdir()}
    if archive_changed:
        archive = next(iter(before))
        changed = lab.filesystem.read_json(archive)
        changed["unit"] = "conflicting legacy proof"
        lab.filesystem.write_json(archive, changed)
        before = {p: lab.filesystem.read_bytes(p) for p in archives.iterdir()}
    code, result = lab.run("index", unit)
    assert code == int(archive_changed), result
    if archive_changed:
        assert "conflicts" in result["results"][0]["error"]
        assert (
            lab.filesystem.read_json(lab.engine.manifest_path(unit))["kind"]
            == "checkpoint"
        )
    assert before == {p: lab.filesystem.read_bytes(p) for p in archives.iterdir()}


def test_orphan_stage_requires_explicit_archival_before_daily_cycle(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    lab.age(unit, 1)
    orphan = (lab.config.mirror / unit.path).with_name(
        ".storage-tier-stage-legacy-orphan"
    )
    orphan.mkdir()
    (orphan / "unverified.bin").write_bytes(b"unrecorded orphan payload")
    identity = (orphan.stat().st_dev, orphan.stat().st_ino)
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 1 and "unclaimed storage artifacts" in report["error"], report
    archive = lab.config.mirror / ".storage-tier/manual-recovery" / orphan.name
    lab.engine.ensure_parent(archive.parent, lab.config.mirror)
    lab.filesystem.rename(orphan, archive)
    assert identity == (archive.stat().st_dev, archive.stat().st_ino)
    assert lab.engine.run("recover", [], dry_run=False)[0] == 0
    assert lab.run("verify", unit)[0] == 0
    lab.filesystem.write_json(
        lab.config.state / "active.json",
        {
            "schema_version": 1,
            "repository": str(lab.config.repo),
            "mirror_mount": str(lab.config.mount),
        },
    )
    assert lab.engine.run("cycle", [], dry_run=False)[0] == 0
    assert (
        lab.filesystem.read_bytes(archive / "unverified.bin")
        == b"unrecorded orphan payload"
    )


def test_e_resident_worktree_outputs_are_never_storage_units(lab: Lab) -> None:
    """Through the link D: and E: name the same bytes; none may be evicted."""
    import dataclasses

    config = dataclasses.replace(lab.config, worktree_outputs_on_mirror=True)
    main = lab.unit("outputs/task/train/experiment/run-001")
    root = config.mirror / ".worktree-outputs"
    run = root / "worker/task/evaluate/experiment/run-002"
    run.mkdir(parents=True)
    (run / "a.bin").write_bytes(b"only copy\n")
    (config.repo / ".worktree-outputs").symlink_to(root, target_is_directory=True)
    local = lab.layout.discover(config, config.repo)
    assert local.units == [main] and local.errors == []
    archive = lab.layout.discover(config, config.mirror)
    assert archive.units == [] and archive.errors == []
    assert not any(".worktree-outputs" in path for path in archive.unmanaged)
    engine = lab.module.Engine(
        config,
        inspector=lab.usage.Inspector(lab.proc),
        mount_check=lambda config: None,
        clock=lambda: lab.now,
    )
    for operation in ("mirror", "evict", "hydrate"):
        try:
            code, _ = engine.run(operation, [".worktree-outputs/worker"], dry_run=False)
        except lab.filesystem.StorageError:
            code = 1
        assert code != 0
    assert (run / "a.bin").read_bytes() == b"only copy\n"
    (config.repo / ".worktree-outputs").unlink()
    (config.repo / ".worktree-outputs").symlink_to(
        config.repo, target_is_directory=True
    )
    assert lab.layout.discover(config, config.repo).errors
