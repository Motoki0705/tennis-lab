"""Configuration checks for the isolated local GPU Actions runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import selectors
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from src.utils.training_queue import RETENTION_NS

ROOT = Path(__file__).parents[3]
WORKFLOWS = ROOT / ".github/workflows"
SCRIPTS = ROOT / "scripts/github_actions"
EXPECTED_LABELS = [
    "self-hosted",
    "linux",
    "x64",
    "local-gpu",
    "cuda",
    "wsl2",
    "tennis-lab",
]


def _load_workflow(name: str) -> tuple[dict[str | bool, Any], str]:
    path = WORKFLOWS / name
    text = path.read_text(encoding="utf-8")
    return cast(dict[str | bool, Any], yaml.safe_load(text)), text


def test_local_gpu_workflows_are_manual_owner_gated_and_read_only() -> None:
    concurrency_groups: set[str] = set()
    for filename, job_name in (
        ("local-gpu-tests.yml", "cuda-tests"),
        ("local-gpu-training.yml", "enqueue"),
    ):
        workflow, text = _load_workflow(filename)
        job = workflow["jobs"][job_name]

        assert "pull_request:" not in text
        assert "pull_request_target:" not in text
        assert "push:" not in text
        assert "workflow_dispatch:" in text
        assert workflow["permissions"] == {"contents": "read"}
        assert job["environment"] == "local-gpu"
        assert job["runs-on"] == EXPECTED_LABELS
        assert "github.actor == github.repository_owner" in job["if"]
        assert "vars.LOCAL_GPU_ACTIONS_ENABLED == 'true'" in job["if"]
        concurrency_groups.add(workflow["concurrency"]["group"])

        checkout = job["steps"][0]
        assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
        assert checkout["with"]["persist-credentials"] is False

    assert concurrency_groups == {"tennis-lab-local-gpu"}


def test_cuda_workflow_runs_serial_marker_tests_under_shared_lock() -> None:
    workflow, text = _load_workflow("local-gpu-tests.yml")

    assert "git submodule update --init --depth 1 third_party/DINO" in text
    assert "TENNIS_LAB_BUILD_CUDA_OPS=1" in text
    assert "python setup.py build_ext --inplace" in text
    assert "torch.cuda.is_available()" in text
    assert "flock --exclusive --timeout 5" in text
    assert "spin test --all --serial -m cuda --no-cov" in text
    assert "/var/lib/tennis-lab-actions/gpu.lock" in text
    assert "-p scripts.ci_metrics.require_execution" in text
    assert "export TENNIS_RUN_B00_RENDERED_ALIGNMENT_GPU_TEST=1" in text
    assert "export PLAYER_DETECTION_GPU_TEST=1" in text
    assert workflow[True]["workflow_dispatch"]["inputs"]["target"]["options"] == [
        "cuda",
        "local-data",
    ]
    local = next(
        step
        for step in workflow["jobs"]["cuda-tests"]["steps"]
        if step["name"] == "Run explicit local-data CPU acceptance"
    )
    assert local["if"] == "inputs.target == 'local-data'"
    assert local["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "local_data and not cuda" in local["run"]
    assert '"$ACCEPTANCE_TEST_FILE"' in local["run"]
    assert "-p scripts.ci_metrics.require_execution" in local["run"]


def test_training_workflow_passes_dispatch_input_through_environment() -> None:
    workflow, text = _load_workflow("local-gpu-training.yml")
    enqueue_step = workflow["jobs"]["enqueue"]["steps"][1]

    assert enqueue_step["run"] == "bash scripts/github_actions/enqueue_training.sh"
    assert enqueue_step["env"]["TRAINING_COMMAND"] == "${{ inputs.command }}"
    assert enqueue_step["env"]["TRAINING_RESOURCE"] == "${{ inputs.resource }}"
    assert workflow[True]["workflow_dispatch"]["inputs"]["resource"] == {
        "description": "Logical GPU reservation (half does not enforce a VRAM cap)",
        "required": False,
        "default": "all",
        "type": "choice",
        "options": ["all", "half"],
    }
    assert "run: ${{ inputs.command }}" not in text


def test_runner_scripts_parse_and_install_security_boundaries() -> None:
    bash_scripts = [
        SCRIPTS / "enqueue_training.sh",
        SCRIPTS / "install_self_hosted_runner.sh",
        SCRIPTS / "install_queue_bundle.sh",
        SCRIPTS / "wsl_keepalive.sh",
        SCRIPTS / "mount_assets.sh",
    ]
    for script in bash_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)

    installer = (SCRIPTS / "install_self_hosted_runner.sh").read_text(encoding="utf-8")
    assert 'readonly RUNNER_USER="tennis-actions"' in installer
    assert "ProtectHome=true" in installer
    assert "for mount_path in /mnt/?; do" in installer
    assert "InaccessiblePaths=$inaccessible_paths" in installer
    assert "InaccessiblePaths=/mnt\n" not in installer
    assert 'mountpoint --quiet "$asset_dir"' in installer
    assets = (SCRIPTS / "mount_assets.sh").read_text()
    assert "mount --options remount,bind,ro" in assets
    assert '"$SCRIPT_DIR/asset_sources.py"' in installer
    assert '"$TOOL_ROOT/bin/asset_paths.conf"' in installer
    assert "Requires=tennis-data-mount.service" in installer
    assert "TRAINING_QUEUE_LOCK_FILE=" in installer
    for suffix in (".gate", ".slot-0", ".slot-1"):
        assert f'"${{GPU_LOCK_FILE}}{suffix}"' in installer
    assert 'chown "$RUNNER_USER:$RUNNER_GROUP" "$lock_path"' in installer
    assert 'chmod 0600 "$lock_path"' in installer
    assert 'runuser -u "$RUNNER_USER" -- test -w "$lock_path"' in installer
    assert (
        "ReadWritePaths=$RUNNER_HOME $STATE_ROOT/runs $STATE_ROOT/training-queue "
        "$GPU_LOCK_FILE $GPU_GATE_FILE $GPU_SLOT_0_FILE $GPU_SLOT_1_FILE" in installer
    )
    assert "KillMode=control-group" in installer
    assert "KillSignal=SIGTERM" in installer
    assert "SendSIGKILL=no" in installer
    assert "TimeoutStopSec=infinity" in installer
    assert "sha256sum --check --status" in installer
    assert 'bash "$SCRIPT_DIR/install_queue_bundle.sh" "$TOOL_ROOT/bin"' in installer
    assert 'readonly CAPACITY_STATE_DIR="/run/tennis-lab-d-capacity"' in installer
    assert "d $CAPACITY_STATE_DIR 0755 $watchdog_uid root -" in installer
    assert 'systemd-tmpfiles --create "$CAPACITY_TMPFILES"' in installer
    assert "Environment=TRAINING_QUEUE_CAPACITY_STATUS_FILE=$CAPACITY_STATE_DIR/status.json" in installer
    assert "ReadOnlyPaths=$STATE_ROOT/assets -$CAPACITY_STATE_DIR" in installer

    enqueue = (SCRIPTS / "enqueue_training.sh").read_text(encoding="utf-8")
    assert 'TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue"' in enqueue
    assert 'readonly EXPECTED_REPOSITORY="Motoki0705/tennis-lab"' in enqueue
    assert "submodule update --init --recursive --depth 1" in enqueue
    assert 'TRAINING_RESOURCE="${TRAINING_RESOURCE:-all}"' in enqueue
    assert 'case "$TRAINING_RESOURCE" in' in enqueue
    assert "TRAINING_RESOURCE must be half or all." in enqueue
    assert 'queue_args+=(--issue "$TRAINING_ISSUE")' in enqueue
    assert '--resource "$TRAINING_RESOURCE"' in enqueue
    assert "- Logical GPU resource: \`$TRAINING_RESOURCE\`" in enqueue


def test_runner_asset_sources_stay_on_configured_mirror_after_d_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
    Config = importlib.import_module("storage_tier.layout").Config

    from scripts.github_actions.asset_sources import asset_sources

    repo, mount = tmp_path / "D/repo", tmp_path / "configured E"
    repo.mkdir(parents=True)
    config_file = tmp_path / "storage.toml"
    config_file.write_text(
        f"mirror_mount = {json.dumps(str(mount))}\ndataset_containers=[]\ndataset_units=[]\n"
    )
    mount.mkdir()
    config = Config.load(repo, config_file)
    for name in ("data", "ckpt"):
        (config.mirror / name).mkdir(parents=True)
        (repo / name).symlink_to(config.mirror / name, target_is_directory=True)
    before = asset_sources(config)
    for name in ("data", "ckpt"):
        (repo / name).unlink()
        (repo / name).mkdir()
    (repo / "data/d-only").write_bytes(b"not yet mirrored")
    after = asset_sources(config)
    assert before == after
    assert after["SOURCE_DATA"] == str(config.mirror / "data")
    assert after["SOURCE_CKPT"] == str(config.mirror / "ckpt")
    assert not (Path(after["SOURCE_DATA"]) / "d-only").exists()


@pytest.mark.parametrize(
    "invalid",
    [None, "root", "device", "writable", "renumbered", "source-root", "unmounted"],
)
def test_assets_runtime_checks_filesystem_root_instead_of_source_path(
    tmp_path: Path,
    invalid: str | None,
) -> None:
    runtime = tmp_path / "mount_assets.sh"
    shutil.copyfile(SCRIPTS / "mount_assets.sh", runtime)
    values = {
        "TARGET_ROOT": str(tmp_path / "assets"),
        "MIRROR_MOUNT": "/mnt/authoritative",
    }
    for name in ("DATA", "CKPT"):
        values.update(
            {
                f"SOURCE_{name}": f"/mnt/authoritative/repo/{name.lower()}",
            }
        )
    (tmp_path / "asset_paths.conf").write_text(
        "".join(f"{k}={shlex.quote(v)}\n" for k, v in values.items())
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "mountpoint").write_text(
        "#!/bin/sh\n"
        + (
            '[ "$2" != "/mnt/authoritative" ]\n'
            if invalid == "unmounted"
            else "exit 0\n"
        )
    )
    (binary / "mountpoint").chmod(0o755)
    finder = binary / "findmnt"
    finder.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nname=sys.argv[-1].rsplit('/',1)[-1]\n"
        + f"invalid={invalid!r}\n"
        + "source=sys.argv[-1].startswith('/mnt/authoritative/')\n"
        + "root='/volume' if invalid=='source-root' else ''\ndevice='8:99' if invalid=='renumbered' else '8:48'\n"
        + "value={'fsroot':root or '/','target':'/mnt/authoritative','maj:min':device} if source else {'fsroot':'/wrong' if invalid=='root' else root+'/repo/'+name,'maj:min':'8:64' if invalid=='device' else device,'options':'rw' if invalid=='writable' else 'ro,noatime'}\n"
        + "print(json.dumps({'filesystems':[value]}))\n"
    )
    finder.chmod(0o755)
    result = subprocess.run(
        ["bash", str(runtime), "start"],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == (
        1 if invalid in {"root", "device", "writable", "unmounted"} else 0
    ), result.stderr


def test_installed_queue_runs_outside_repository_with_system_python_only(
    tmp_path: Path,
) -> None:
    # The installer itself needs root and touches system services. Exercise its
    # exact runtime-copy helper instead, outside Git even with --basetemp in a
    # worktree. No venv, PYTHONPATH, site packages or user Python configuration.
    with tempfile.TemporaryDirectory(
        prefix="training-queue-install-", dir="/tmp"
    ) as temporary:
        root = Path(temporary).resolve()
        assert not root.is_relative_to(ROOT.resolve())
        binary_directory = root / "opt/tennis-lab-actions/bin"
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(root),
            "TRAINING_QUEUE_DIR": str(tmp_path / "queue"),
            "TRAINING_QUEUE_LOCK_FILE": str(tmp_path / "gpu.lock"),
            "TRAINING_QUEUE_SYSTEM_LOCK_FILE": str(tmp_path / "unprovisioned/gpu.lock"),
            "GIT_CEILING_DIRECTORIES": str(root),
            "TRAINING_QUEUE_TEST_NOW_NS": "1792368000000000000",
        }
        subprocess.run(
            ["bash", str(SCRIPTS / "install_queue_bundle.sh"), str(binary_directory)],
            env=environment,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        def run(*arguments: str) -> str:
            return subprocess.run(
                ["bash", str(binary_directory / "training_queue.sh"), *arguments],
                env=environment,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        # Disable third-party packages for every helper call, and record which
        # interpreter was used without depending on the test runner's Python.
        shim = root / "system-python"
        shim.mkdir()
        calls = root / "python-calls"
        (shim / "python3").write_text(
            f'#!/bin/sh\necho "$*" >> "{calls}"\nexec /usr/bin/python3 -I -S "$@"\n'
        )
        (shim / "python3").chmod(0o755)
        environment["PATH"] = f"{shim}:/usr/bin:/bin"
        bundle = {
            path.relative_to(binary_directory): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in binary_directory.rglob("*")
            if path.is_file()
        }
        for path in binary_directory.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        binary_directory.chmod(0o555)
        try:
            job = (
                run(
                    "add",
                    "true",
                    "--name",
                    "installed",
                    "--idempotency-key",
                    "installed-attempt",
                )
                .strip()
                .removeprefix("queued: ")
            )
            assert "queued=1" in run("status")
            assert run("cancel", job).strip() == "cancelled"
            queue = tmp_path / "queue"
            completion = queue / "control/completions" / f"{job[:-4]}.json"
            record = json.loads(completion.read_text())
            log = queue / "logs" / f"{job[:-4]}.log"
            log.write_text("retained then collected")
            environment["TRAINING_QUEUE_TEST_NOW_NS"] = str(
                record["finished_at_ns"] + RETENTION_NS
            )
            assert "cancelled=1" in run("status")
            assert log.exists()
            environment["TRAINING_QUEUE_TEST_NOW_NS"] = str(
                record["finished_at_ns"] + RETENTION_NS + 1
            )
            assert "cancelled=0" in run("status")
            assert not log.exists()
            assert not (queue / "cancelled" / job).exists()
            assert "history=expired" in run("list", "--include-expired")
            assert (
                run(
                    "add",
                    "true",
                    "--name",
                    "installed",
                    "--idempotency-key",
                    "installed-attempt",
                ).strip()
                == f"queued: {job}"
            )
            assert all(
                line.startswith("-B ") for line in calls.read_text().splitlines()
            )
            assert {
                path.relative_to(binary_directory): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in binary_directory.rglob("*")
                if path.is_file()
            } == bundle
            assert not list(binary_directory.rglob("__pycache__"))
        finally:
            for path in binary_directory.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            binary_directory.chmod(0o755)


def test_installed_ci_queue_holds_job_until_fresh_watchdog_verdict(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "installed"
    subprocess.run(
        ["bash", str(SCRIPTS / "install_queue_bundle.sh"), str(bin_dir)], check=True
    )
    queue = tmp_path / "queue"
    status = tmp_path / "watchdog-status.json"
    done = tmp_path / "payload"
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TRAINING_QUEUE_DIR": str(queue),
        "TRAINING_QUEUE_LOCK_FILE": str(tmp_path / "gpu.lock"),
        "TRAINING_QUEUE_CAPACITY_STATUS_FILE": str(status),
        "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
    }
    command = ["bash", str(bin_dir / "training_queue.sh")]
    result = subprocess.run(
        [*command, "add", f"touch {shlex.quote(str(done))}"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    job = result.stdout.strip().removeprefix("queued: ")
    worker = subprocess.Popen(
        [*command, "serve", "--idle-timeout", "0"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert worker.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(worker.stderr, selectors.EVENT_READ)
            assert selector.select(timeout=5)
            assert "D: watchdog status is missing" in worker.stderr.readline()
        assert (queue / "jobs" / job).is_file()
        assert not done.exists()

        def publish(available: int, age_seconds: int, decision: str) -> None:
            temporary = status.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "measured_at_ns": time.time_ns() - age_seconds * 10**9,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        "boottime_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                        - age_seconds * 10**9,
                        "available_bytes": available,
                        "threshold_bytes": 21_474_836_480,
                        "decision": decision,
                        "error": None,
                    }
                )
            )
            temporary.replace(status)

        for available, age, decision, diagnostic in [
            (21_474_836_480, 91, "allow", "older than 90 seconds"),
            (21_474_836_479, 0, "blocked", "below the"),
        ]:
            publish(available, age, decision)
            verdict = subprocess.run(
                ["python3", "-I", "-S", str(bin_dir / "d_capacity.py"), "--state-file", str(status)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            assert verdict.returncode == 75
            assert diagnostic in verdict.stderr
            assert (queue / "jobs" / job).is_file()
            assert not done.exists()
        publish(21_474_836_480, 0, "allow")
        stdout, stderr = worker.communicate(timeout=8)
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.communicate(timeout=8)
    assert worker.returncode == 0, (stdout, stderr)
    assert done.is_file() and (queue / "done" / job).is_file()
