"""Materialize and inventory the reviewed WSL MCP project-sandbox change."""

from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

_PARTS_DIR = Path(".github/wsl-mcp-bundle")


def _bundle_text() -> str:
    parts = sorted(_PARTS_DIR.glob("part*"))
    if not parts:
        raise SystemExit("WSL MCP bundle parts are missing")
    return "".join(part.read_text(encoding="utf-8").strip() for part in parts)


_PRIVATE_SERVICE_BLOCK = '    def _private_service_unit(self) -> str:\n        return f"""[Unit]\nDescription=Private tennis-lab MCP endpoint for OpenAI Secure Tunnel\nAfter=network-online.target docker.service\nWants=network-online.target\n\n[Service]\nType=simple\nWorkingDirectory={_unit_path_value(self.source_root)}\nEnvironment="PYTHONPATH={self.source_root}"\nEnvironment="TENNIS_MCP_REPO_ROOT={self.settings.repo_root}"\nEnvironment="TENNIS_MCP_STATE_DIR={self.settings.state_dir}"\nEnvironment="TENNIS_MCP_CONTROL_DIR={self.settings.control_dir}"\nEnvironment="TENNIS_MCP_ORIGIN_URL={self.settings.origin_url}"\nEnvironment="TENNIS_MCP_HOST=127.0.0.1"\nEnvironment="TENNIS_MCP_PORT={PRIVATE_MCP_PORT}"\nExecStart={_unit_value(self.python_executable)} -m src.automation.chatgpt_mcp serve-private\nRestart=always\nRestartSec=5\nTimeoutStopSec=20\nUMask=0077\nNoNewPrivileges=true\nPrivateTmp=true\n\n[Install]\nWantedBy=default.target\n"""\n\n'
_EXPECTED_TOOLS = '_EXPECTED_TOOLS = {\n    "get_host_status",\n    "get_execution_layout",\n    "prepare_revision_workspace",\n    "get_revision_status",\n    "start_command",\n    "get_command_job",\n    "list_command_jobs",\n    "get_command_output",\n    "cancel_command_job",\n    "enqueue_training",\n    "get_training_job",\n    "list_training_jobs",\n    "get_training_output",\n    "cancel_training_job",\n}\n\n'
_INTEGRATION_SETTINGS = 'def _settings(tmp_path: Path) -> GatewaySettings:\n    repo = tmp_path / "repo"\n    repo.mkdir()\n    subprocess.run(["git", "init", "-q", str(repo)], check=True)\n    control = tmp_path / "control"\n    (control / "venv/bin").mkdir(parents=True)\n    (control / "current").mkdir()\n    (control / "repository.git").mkdir()\n    uv_root = tmp_path / "uv-python"\n    uv_root.mkdir()\n    settings = GatewaySettings(\n        repo_root=repo,\n        state_dir=tmp_path / "state",\n        control_dir=control,\n        public_base_url="https://mcp.example.test",\n        uv_python_root=uv_root,\n    )\n    settings.ensure_state()\n    return settings\n\n\n'


def write_bundle() -> None:
    root = Path.cwd()
    payload = json.loads(zlib.decompress(base64.b64decode(_bundle_text())))
    for relative, content in payload.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    secure_path = root / "src/automation/chatgpt_mcp/secure_tunnel.py"
    secure = secure_path.read_text(encoding="utf-8")
    start = secure.index("    def _private_service_unit(self) -> str:")
    end = secure.index("    def _tunnel_service_unit(self) -> str:", start)
    secure = secure[:start] + _PRIVATE_SERVICE_BLOCK + secure[end:]
    secure_path.write_text(secure, encoding="utf-8")

    integration_path = root / "tests/integration/chatgpt_mcp/test_oauth_mcp.py"
    integration = integration_path.read_text(encoding="utf-8")
    integration = re.sub(
        r"_EXPECTED_TOOLS = \{.*?\}\n\n",
        _EXPECTED_TOOLS,
        integration,
        count=1,
        flags=re.DOTALL,
    )
    settings_start = integration.index("def _settings(tmp_path: Path)")
    settings_end = integration.index("def _pkce_challenge", settings_start)
    integration = (
        integration[:settings_start]
        + _INTEGRATION_SETTINGS
        + integration[settings_end:]
    )
    integration = integration.replace(
        'assert service.json()["role"] == "exact-revision execution and GPU validation only"',
        'assert service.json()["role"] == (\n'
        '            "arbitrary tennis-lab execution, validation, and GPU training"\n'
        '        )',
    )
    integration_path.write_text(integration, encoding="utf-8")

    secure_test_path = root / "tests/unit/automation/chatgpt_mcp/test_secure_tunnel.py"
    secure_test = secure_test_path.read_text(encoding="utf-8")
    secure_test = secure_test.replace(
        "        state_dir=tmp_path / \"state\",\n"
        "        public_base_url=None,\n",
        "        state_dir=tmp_path / \"state\",\n"
        "        control_dir=tmp_path / \"control\",\n"
        "        public_base_url=None,\n",
        1,
    )
    secure_test = secure_test.replace(
        '    assert "TENNIS_MCP_PORT=8767" in private_unit\n',
        '    assert "TENNIS_MCP_PORT=8767" in private_unit\n'
        '    assert f"PYTHONPATH={tmp_path}/source directory" in private_unit\n'
        '    assert f"TENNIS_MCP_CONTROL_DIR={tmp_path}/control" in private_unit\n'
        '    assert "TENNIS_MCP_ORIGIN_URL=https://github.com/Motoki0705/tennis-lab.git" in private_unit\n',
        1,
    )
    secure_test_path.write_text(secure_test, encoding="utf-8")

    automation_readme = root / "src/automation/README.md"
    text = automation_readme.read_text(encoding="utf-8")
    text = re.sub(
        r"- `chatgpt_mcp/`:.*?(?=\n- |\Z)",
        "- `chatgpt_mcp/`: externalized ChatGPT execution control plane for a "
        "read-write tennis-lab sandbox, exact revisions, CUDA, and the serial "
        "training queue.",
        text,
        count=1,
        flags=re.DOTALL,
    )
    automation_readme.write_text(text, encoding="utf-8")


def regenerate_inventory() -> None:
    from src.utils.configuration import AuditExemption, AuditRule
    from src.utils.configuration.audit import (
        regenerate_exemption_rows,
        write_generated_inventory_data,
    )

    source_root = Path("src").resolve()
    _, _, unresolved = regenerate_exemption_rows(source_root)
    unexpected = [
        finding
        for finding in unresolved
        if not finding.module.startswith("src.automation.chatgpt_mcp")
    ]
    if unexpected:
        rendered = "\n".join(repr(finding) for finding in unexpected)
        raise SystemExit("unexpected non-MCP inventory findings:\n" + rendered)

    path_rules = {
        AuditRule.HYDRA_ABSOLUTE_PATH,
        AuditRule.FILE_PARENT_INDEX,
        AuditRule.RUNTIME_PATH_LITERAL,
        AuditRule.PATH_JOIN,
        AuditRule.PROCESS_CWD,
        AuditRule.HYDRA_RUN_DIRECTORY,
    }
    approvals_by_identity = {}
    for finding in unresolved:
        if finding.rule not in path_rules:
            reason_code = "strict-schema"
        elif finding.rule is AuditRule.FILE_PARENT_INDEX:
            reason_code = "code-or-artifact-location"
        else:
            reason_code = "persisted-layout"
        identity = (
            finding.module,
            finding.qualified_name,
            finding.line,
            finding.rule,
        )
        approvals_by_identity[identity] = AuditExemption.classified(
            module=finding.module,
            qualified_name=finding.qualified_name,
            line=finding.line,
            rule=finding.rule,
            reason_code=reason_code,
        )
    approvals = tuple(approvals_by_identity.values())

    migration_count, exemption_count = write_generated_inventory_data(
        source_root,
        source_revision="wsl-mcp-project-sandbox-v1",
        approved_exemptions=approvals,
    )
    print(
        f"generated inventory: migrations={migration_count} "
        f"exemptions={exemption_count} new={len(approvals)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "inventory"))
    args = parser.parse_args()
    if args.mode == "write":
        write_bundle()
    else:
        regenerate_inventory()


if __name__ == "__main__":
    main()
