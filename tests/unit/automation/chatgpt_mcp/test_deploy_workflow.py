from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_WORKFLOW = _ROOT / ".github/workflows/deploy-wsl-mcp.yml"
_STATUS_WORKFLOW = _ROOT / ".github/workflows/publish-wsl-mcp-deploy-status.yml"
_TRUSTED_RUNNER_INSTALLER = (
    _ROOT / "scripts/github_actions/install_trusted_mcp_deploy_runner.sh"
)
_TRUSTED_RUNNER_HOOK = (
    _ROOT / "scripts/github_actions/authorize_trusted_mcp_deploy_job.sh"
)


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _status_workflow_text() -> str:
    return _STATUS_WORKFLOW.read_text(encoding="utf-8")


def test_deploy_auto_runs_owner_main_changes_without_environment_approval() -> None:
    text = _workflow_text()

    assert "push:" in text
    assert "      - main" in text
    assert "workflow_dispatch:" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.workflow_ref == format(" in text
    assert "github.actor == github.repository_owner" in text
    assert "vars.LOCAL_GPU_ACTIONS_ENABLED == 'true'" in text
    assert "environment: local-gpu" not in text


def test_deploy_uses_external_exact_revision_control_plane() -> None:
    text = _workflow_text()

    assert 'runs-on: [self-hosted, linux, x64, trusted-mcp-deploy]' in text
    assert 'persist-credentials: false' in text
    assert 'venv_key="$(cat pyproject.toml uv.lock | sha256sum' in text
    assert 'UV_PROJECT_ENVIRONMENT="$venv_target"' in text
    assert '--source-root "$GITHUB_WORKSPACE"' in text
    assert '--expected-sha "$GITHUB_SHA"' in text
    assert '--reuse-existing-key' in text
    assert 'TENNIS_MCP_GPU_LOCK_FILE="$GPU_LOCK_FILE"' in text
    assert 'test "$(cat "$MCP_CONTROL_DIR/runtime-version")" = "$GITHUB_SHA"' in text


def test_trusted_deploy_runner_has_local_authorization_and_service_boundaries() -> None:
    installer = _TRUSTED_RUNNER_INSTALLER.read_text(encoding="utf-8")
    hook = _TRUSTED_RUNNER_HOOK.read_text(encoding="utf-8")

    assert 'EXPECTED_USER="kamimura"' in installer
    assert 'RUNNER_LABELS="wsl2,tennis-lab,trusted-mcp-deploy"' in installer
    assert "ACTIONS_RUNNER_HOOK_JOB_STARTED=" in installer
    assert "ProtectHome=read-only" in installer
    assert "ProtectSystem=strict" in installer
    assert "ReadWritePaths=$MCP_STATE_DIR" in installer
    assert "ReadWritePaths=$MCP_CONTROL_DIR" in installer
    assert "ReadWritePaths=$PROJECT_ROOT" in installer
    assert "ReadOnlyPaths=$runner_env" in installer
    assert "ReadOnlyPaths=$HOOK_PATH" in installer
    assert "ExecStart=/bin/bash $RUNNER_ROOT/run.sh" in installer
    assert 'systemctl --user enable "$SERVICE_NAME"' in installer
    assert 'systemctl --user restart "$SERVICE_NAME"' in installer

    for variable in (
        "GITHUB_REPOSITORY",
        "GITHUB_ACTOR",
        "GITHUB_TRIGGERING_ACTOR",
        "GITHUB_REF",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_WORKFLOW_SHA",
        "GITHUB_JOB",
        "GITHUB_EVENT_NAME",
    ):
        assert variable in hook
    assert "push | workflow_dispatch" in hook
    assert "deploy-wsl-mcp.yml@${EXPECTED_REF}" in hook


def _trusted_hook_environment(tmp_path: Path) -> dict[str, str]:
    revision = "1" * 40
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "after": revision,
                "ref": "refs/heads/main",
                "repository": {"full_name": "Motoki0705/tennis-lab"},
                "sender": {"login": "Motoki0705"},
            }
        ),
        encoding="utf-8",
    )
    return {
        **os.environ,
        "GITHUB_REPOSITORY": "Motoki0705/tennis-lab",
        "GITHUB_ACTOR": "Motoki0705",
        "GITHUB_TRIGGERING_ACTOR": "Motoki0705",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_WORKFLOW": "Deploy WSL MCP",
        "GITHUB_WORKFLOW_REF": (
            "Motoki0705/tennis-lab/.github/workflows/"
            "deploy-wsl-mcp.yml@refs/heads/main"
        ),
        "GITHUB_WORKFLOW_SHA": revision,
        "GITHUB_SHA": revision,
        "GITHUB_JOB": "deploy",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_PATH": str(event_path),
    }


def test_trusted_deploy_runner_hook_accepts_only_expected_job(tmp_path: Path) -> None:
    environment = _trusted_hook_environment(tmp_path)

    accepted = subprocess.run(
        [str(_TRUSTED_RUNNER_HOOK)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "trusted MCP deploy job authorized" in accepted.stdout

    environment["GITHUB_EVENT_NAME"] = "pull_request"
    environment["GITHUB_REF"] = "refs/pull/123/merge"
    rejected = subprocess.run(
        [str(_TRUSTED_RUNNER_HOOK)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert rejected.returncode == 78
    assert "trusted MCP deploy runner rejected job" in rejected.stderr


def test_deploy_verifies_tool_surface_and_host_boundary() -> None:
    text = _workflow_text()

    for tool in (
        "get_host_status",
        "get_execution_layout",
        "prepare_revision_workspace",
        "start_command",
        "enqueue_training",
        "cancel_training_job",
    ):
        assert f'"{tool}"' in text

    assert 'test -w /tennis-lab' in text
    assert 'test ! -e /var/run/docker.sock' in text
    assert 'test ! -e /mnt/c' in text
    assert 'test ! -e /tennis-lab/.git/HEAD' in text
    assert 'test ! -e /workspace/.git' in text
    assert '/home/kamimura/.local/share/tennis-lab-chatgpt-mcp' in text
    assert '/home/kamimura/.local/state/tennis-lab-chatgpt-mcp' in text
    assert 'layout["direct_concurrency"] != 2' in text
    assert 'layout["direct_memory_limit_gb"] != 24' in text
    assert 'layout["queued_memory_limit_gb"] != 48' in text


def test_deploy_runs_cpu_regression_and_serial_cuda_smoke() -> None:
    text = _workflow_text()

    assert 'PYTHONPATH="$GITHUB_WORKSPACE"' in text
    assert 'TMPDIR=/tmp' in text
    assert 'PYTEST_DEBUG_TEMPROOT' not in text
    assert '"$trusted_python" -m pytest -q -n 0' in text
    assert 'tests/unit/automation/chatgpt_mcp' in text
    assert 'tests/integration/chatgpt_mcp' in text
    assert '"python -m pytest -q "' not in text
    assert '"enqueue_training"' in text
    assert 'assert torch.cuda.is_available()' in text
    assert 'mcp-deploy-cuda-smoke' in text


def test_deploy_status_is_published_by_separate_github_hosted_workflow() -> None:
    deploy = _workflow_text()
    status = _status_workflow_text()

    assert "statuses: write" not in deploy
    assert "workflow_run:" in status
    assert "- Deploy WSL MCP" in status
    assert "- requested" in status
    assert "- completed" in status
    assert "statuses: write" in status
    assert "runs-on: ubuntu-24.04" in status
    assert '"context": "wsl-mcp/deploy"' in status
    assert 'state = "pending"' in status
    assert '"success": "success"' in status
    assert '"failure": "failure"' in status
    assert "github.event.workflow_run.head_sha" in status
    assert "github.event.workflow_run.html_url" in status
