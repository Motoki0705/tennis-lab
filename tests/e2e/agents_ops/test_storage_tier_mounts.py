"""Real bind aliases in a private namespace must not certify or delete checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture(scope="module")
def private_mount_namespace(tmp_path_factory: pytest.TempPathFactory) -> None:
    if shutil.which("unshare") is None or shutil.which("mount") is None:
        pytest.skip("Linux util-linux unshare/mount are unavailable")
    directory = tmp_path_factory.mktemp("private-bind-probe")
    probe = subprocess.run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",
            sys.executable,
            "-c",
            "import subprocess,sys; subprocess.run(['mount','--bind',sys.argv[1],sys.argv[1]],check=True)",
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode:
        pytest.skip(
            f"private user/mount namespaces unavailable: {probe.stderr.strip()}"
        )


SCRIPT = r"""
import json, os, subprocess, sys, time
from pathlib import Path
import storage_tier.engine as engine_module
import storage_tier.layout as layout_module
from storage_tier.engine import Engine
from storage_tier.filesystem import read_bytes
from storage_tier.layout import Config, Unit
from storage_tier.usage import Inspector

root, tier, position, guard = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
repo, mount, proc = root/'D/repo', root/'E', root/'proc'
repo.mkdir(parents=True); mount.mkdir(); proc.mkdir()
config = Config(repo, mount, ('data/task',))
config.mirror.mkdir(parents=True)
base = repo if tier == 'D' else config.mirror
payload = base/'ckpt/v1/checkpoint.bin'
payload.parent.mkdir(parents=True)
payload.write_bytes(b'checkpoint must survive')
old = int((time.time()-15*86400)*10**9)
os.utime(payload, ns=(old, payload.stat().st_mtime_ns))
if position == 'repo':
    (base/'data/task/v1').mkdir(parents=True)
    (base/'data/task/v1/dataset.bin').write_bytes(b'a real independent dataset')
    source = target = base
else:
    target = base/'data/task' if position == 'ancestor' else base/'data/task/v1'
    target.mkdir(parents=True)
    source = base/'ckpt' if position == 'ancestor' else base/'ckpt/v1'
subprocess.run(['mount', '--bind', str(source), str(target)], check=True)
if guard == 'identity-only':
    # Independent defense: bypass path-boundary checks only in this child.
    engine_module.real_directory = lambda *args: None
    layout_module.real_directory = lambda *args: None
engine = Engine(config, inspector=Inspector(proc), mount_check=lambda config: None)
try:
    code, report = engine.run('cycle' if tier == 'D' else 'index', ['data/task/v1'], dry_run=False)
except Exception as exc:
    _, scan = engine.run('scan', [], dry_run=True)
    code, report = 1, {'error': str(exc), 'layout_errors': scan['layout_errors']}
print(json.dumps({
    'code': code, 'report': report,
    'checkpoint_survived': payload.exists() and read_bytes(payload) == b'checkpoint must survive',
    'dataset_certified': engine.manifest_path(Unit('data/task/v1', 'dataset')).exists(),
}))
"""


@pytest.mark.parametrize(
    "tier,position,guard",
    [
        ("D", "ancestor", "normal"),
        ("D", "root", "normal"),
        ("E", "ancestor", "normal"),
        ("E", "root", "normal"),
        ("E", "repo", "normal"),
        ("D", "ancestor", "identity-only"),
        ("E", "ancestor", "identity-only"),
    ],
)
def test_checkpoint_bind_alias_is_rejected_before_certification(
    tmp_path: Path, private_mount_namespace: None, tier: str, position: str, guard: str
) -> None:
    environment = dict(os.environ, PYTHONPATH=str(OPS))
    result = subprocess.run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",
            sys.executable,
            "-c",
            SCRIPT,
            str(tmp_path),
            tier,
            position,
            guard,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["code"] == 1, observed
    assert not observed["dataset_certified"], observed
    assert observed["checkpoint_survived"], observed
    expected_reason = (
        "checkpoint directory alias" if guard == "identity-only" else "mount"
    )
    assert expected_reason in json.dumps(observed["report"]), observed


def test_external_file_bind_reference_is_inspected_and_survives_unlink(
    tmp_path: Path, private_mount_namespace: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl
    import importlib
    import select

    monkeypatch.syspath_prepend(str(OPS))
    filesystem = importlib.import_module("storage_tier.filesystem")
    leases = importlib.import_module("storage_tier.leases")
    usage = importlib.import_module("storage_tier.usage")
    unit, proc = tmp_path / "unit", tmp_path / "proc"
    unit.mkdir()
    proc.mkdir()
    source, alias = unit / "payload.bin", tmp_path / "external alias.bin"
    source.write_bytes(b"apple\n")
    alias.touch()
    tree = filesystem.snapshot(unit)
    script = r"""
import json, os, subprocess, sys
from pathlib import Path
source, alias = Path(sys.argv[1]), Path(sys.argv[2])
print('waiting-before-bind', flush=True)
sys.stdin.readline()
subprocess.run(['mount','--bind',str(source),str(alias)], check=True)
info = alias.stat()
point = str(alias).replace('\\','\\134').replace(' ','\\040')
line = next(line for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4] == point)
print(json.dumps({'pid':os.getpid(),'dev':info.st_dev,'ino':info.st_ino,'nlink':info.st_nlink,'mountinfo':line}), flush=True)
sys.stdin.readline()
fd = os.open(alias, os.O_RDWR | os.O_NOATIME)
os.pwrite(fd,b'pear!\n',0)
print(json.dumps({'nlink':os.fstat(fd).st_nlink,'payload':os.pread(fd,6,0).decode()}), flush=True)
os.close(fd)
"""
    child = subprocess.Popen(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",
            sys.executable,
            "-c",
            script,
            str(source),
            str(alias),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fd = -1
    try:
        assert child.stdout is not None and select.select([child.stdout], [], [], 10)[0]
        assert child.stdout.readline().strip() == "waiting-before-bind"
        (proc / str(child.pid)).symlink_to(
            f"/proc/{child.pid}", target_is_directory=True
        )
        before_reference = usage.Inspector(proc).inspect((unit,), tree)
        assert child.stdin is not None
        child.stdin.write("bind\n")
        child.stdin.flush()
        assert select.select([child.stdout], [], [], 10)[0]
        ready = json.loads(child.stdout.readline())
        inspected = usage.Inspector(proc).inspect((unit,), tree)
        target = Path(f"/proc/{child.pid}/root") / str(alias).lstrip("/")
        visible = target.stat()
        fd = os.open(source, os.O_RDONLY | os.O_NOATIME)
        with leases.write_lease(fd):
            before = fcntl.fcntl(fd, fcntl.F_GETLEASE)
            source.unlink()
            after = fcntl.fcntl(fd, fcntl.F_GETLEASE)
        stdout, stderr = child.communicate("write\n", timeout=10)
        assert child.returncode == 0, stderr
        written = json.loads(stdout)
        observed = {
            "kernel": os.uname().release,
            "reference": "private-file-bind",
            "before_reference_busy": bool(before_reference.busy),
            "initial_nlink": ready["nlink"],
            "same_inode_through_proc_root": (visible.st_dev, visible.st_ino)
            == (ready["dev"], ready["ino"]),
            "existing_reference_busy": bool(inspected.busy),
            "busy": inspected.busy,
            "unknown": inspected.unknown,
            "lease_before_unlink": before,
            "lease_after_unlink": after,
            "reopened_nlink": written["nlink"],
            "reopened_payload": written["payload"],
            "source_exists": source.exists(),
            "mountinfo": ready["mountinfo"],
        }
        print(json.dumps(observed, sort_keys=True))
        assert not before_reference.busy and not before_reference.unknown, observed
        assert observed["same_inode_through_proc_root"] and not inspected.unknown, (
            observed
        )
        assert ready["nlink"] == 1 and before == after == fcntl.F_WRLCK, observed
        assert written == {"nlink": 0, "payload": "pear!\n"} and not source.exists(), (
            observed
        )
        assert inspected.busy and any("mount" in reason for reason in inspected.busy), (
            observed
        )
    finally:
        if fd >= 0:
            os.close(fd)
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize("operation", ["evict", "retire-mirror"])
def test_stacked_file_bind_cannot_hide_reference_from_removal(
    tmp_path: Path,
    private_mount_namespace: None,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import importlib
    import select
    import time

    monkeypatch.syspath_prepend(str(OPS))
    module = importlib.import_module("storage_tier.engine")
    layout = importlib.import_module("storage_tier.layout")
    usage = importlib.import_module("storage_tier.usage")
    repo, mount, proc = tmp_path / "D/repo", tmp_path / "E", tmp_path / "proc"
    repo.mkdir(parents=True)
    mount.mkdir()
    proc.mkdir()
    config = layout.Config(repo, mount, ("data/task",))
    unit = "data/task/v1"
    local = repo / unit
    local.mkdir(parents=True)
    (local / "a.bin").write_bytes(b"apple\n")
    old = int((time.time() - 15 * 86400) * 10**9)
    os.utime(local / "a.bin", ns=(old, (local / "a.bin").stat().st_mtime_ns))
    engine = module.Engine(
        config, inspector=usage.Inspector(proc), mount_check=lambda c: None
    )
    code, report = engine.run("mirror", [unit], dry_run=False)
    assert code == 0, report
    source = (repo if operation == "evict" else config.mirror) / unit / "a.bin"
    identity = source.stat().st_ino
    alias, cover = tmp_path / "alias.bin", tmp_path / "cover.bin"
    alias.touch()
    cover.write_bytes(b"cover\n")
    consumer = r"""
import json, os, subprocess, sys
from pathlib import Path
source,alias,cover=map(Path,sys.argv[1:])
subprocess.run(['mount','--bind',str(source),str(alias)],check=True)
original=alias.stat().st_ino
subprocess.run(['mount','--bind',str(cover),str(alias)],check=True)
print(json.dumps({'pid':os.getpid(),'original':original,'visible':alias.stat().st_ino}),flush=True)
sys.stdin.readline()
print('consumer-done',flush=True)
"""
    child = subprocess.Popen(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",
            sys.executable,
            "-c",
            consumer,
            str(source),
            str(alias),
            str(cover),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and select.select([child.stdout], [], [], 10)[0]
        line = child.stdout.readline()
        assert line, child.stderr.read() if child.stderr else "no consumer output"
        ready = json.loads(line)
        assert ready["original"] == identity and ready["visible"] != identity, ready
        (proc / str(child.pid)).symlink_to(
            f"/proc/{child.pid}", target_is_directory=True
        )
        if operation == "retire-mirror":
            (local / "a.bin").write_bytes(b"peach\n")
        code, report = engine.run(
            "evict" if operation == "evict" else "mirror", [unit], dry_run=False
        )
        stdout, stderr = child.communicate("probe\n", timeout=10)
        assert child.returncode == 0, stderr
        assert stdout.strip() == "consumer-done"
        observed = {
            "code": code,
            "report": report,
            "original_inode": identity,
            "named_original_survives": source.exists()
            and source.stat().st_ino == identity,
        }
        print(json.dumps(observed))
        assert code == 1 and "usage cannot be verified" in json.dumps(report), observed
        assert observed["named_original_survives"], observed
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
