"""Configuration checks for the isolated local GPU Actions runner."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

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


def _load_workflow(name: str) -> tuple[dict[str, Any], str]:
    path = WORKFLOWS / name
    text = path.read_text(encoding="utf-8")
    return cast(dict[str, Any], yaml.safe_load(text)), text


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
    _, text = _load_workflow("local-gpu-tests.yml")

    assert "git submodule update --init --depth 1 third_party/DINO" in text
    assert "TENNIS_LAB_BUILD_CUDA_OPS=1" in text
    assert "python setup.py build_ext --inplace" in text
    assert "torch.cuda.is_available()" in text
    assert "flock --exclusive --timeout 5" in text
    assert "spin test --all --serial -m cuda --no-cov" in text
    assert "/var/lib/tennis-lab-actions/gpu.lock" in text


def test_training_workflow_passes_dispatch_input_through_environment() -> None:
    workflow, text = _load_workflow("local-gpu-training.yml")
    enqueue_step = workflow["jobs"]["enqueue"]["steps"][1]

    assert enqueue_step["run"] == "bash scripts/github_actions/enqueue_training.sh"
    assert enqueue_step["env"]["TRAINING_COMMAND"] == "${{ inputs.command }}"
    assert "run: ${{ inputs.command }}" not in text


def test_runner_scripts_parse_and_install_security_boundaries() -> None:
    bash_scripts = [
        SCRIPTS / "enqueue_training.sh",
        SCRIPTS / "install_self_hosted_runner.sh",
        SCRIPTS / "wsl_keepalive.sh",
    ]
    for script in bash_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)

    installer = (SCRIPTS / "install_self_hosted_runner.sh").read_text(
        encoding="utf-8"
    )
    assert 'readonly RUNNER_USER="tennis-actions"' in installer
    assert 'readonly TRUSTED_MCP_USER="kamimura"' in installer
    assert "ProtectHome=true" in installer
    assert "for mount_path in /mnt/?; do" in installer
    assert "InaccessiblePaths=$inaccessible_paths" in installer
    assert "InaccessiblePaths=/mnt\n" not in installer
    assert 'mountpoint --quiet "$asset_dir"' in installer
    assert "mount --options remount,bind,ro" in installer
    assert "TRAINING_QUEUE_LOCK_FILE=" in installer
    assert '-g "$TRUSTED_MCP_GROUP" -m 0710 "$STATE_ROOT"' in installer
    assert 'chown "$RUNNER_USER:$TRUSTED_MCP_GROUP" "$GPU_LOCK_FILE"' in installer
    assert 'chmod 0660 "$GPU_LOCK_FILE"' in installer
    assert 'runuser -u "$RUNNER_USER" -- test -w "$GPU_LOCK_FILE"' in installer
    assert 'runuser -u "$TRUSTED_MCP_USER" -- test -w "$GPU_LOCK_FILE"' in installer
    assert "sha256sum --check --status" in installer

    enqueue = (SCRIPTS / "enqueue_training.sh").read_text(encoding="utf-8")
    assert 'TRAINING_QUEUE_DIR="$STATE_ROOT/training-queue"' in enqueue
    assert 'readonly EXPECTED_REPOSITORY="Motoki0705/tennis-lab"' in enqueue
    assert "submodule update --init --recursive --depth 1" in enqueue
