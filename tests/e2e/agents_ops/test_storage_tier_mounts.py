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
