from __future__ import annotations

from pathlib import Path


_WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / ".github/workflows/deploy-wsl-mcp.yml"
)


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_deploy_uses_external_exact_revision_control_plane() -> None:
    text = _workflow_text()

    assert 'runs-on: [self-hosted, linux, x64, local-gpu, cuda, wsl2, tennis-lab]' in text
    assert 'persist-credentials: false' in text
    assert 'venv_key="$(cat pyproject.toml uv.lock | sha256sum' in text
    assert 'UV_PROJECT_ENVIRONMENT="$venv_target"' in text
    assert '--source-root "$GITHUB_WORKSPACE"' in text
    assert '--expected-sha "$GITHUB_SHA"' in text
    assert '--reuse-existing-key' in text
    assert 'TENNIS_MCP_GPU_LOCK_FILE="$GPU_LOCK_FILE"' in text
    assert 'test "$(cat "$MCP_CONTROL_DIR/runtime-version")" = "$GITHUB_SHA"' in text


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

    assert 'tests/unit/automation/chatgpt_mcp' in text
    assert 'tests/integration/chatgpt_mcp' in text
    assert '"enqueue_training"' in text
    assert 'assert torch.cuda.is_available()' in text
    assert 'mcp-deploy-cuda-smoke' in text
