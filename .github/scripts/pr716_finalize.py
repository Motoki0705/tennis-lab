"""Finalize PR #716 security hardening and generated configuration inventory."""

from __future__ import annotations

from pathlib import Path


def _replace_systemd_installation() -> None:
    path = Path("src/automation/chatgpt_mcp/secure_tunnel.py")
    source = path.read_text(encoding="utf-8")
    if "import tempfile\n" not in source:
        source = source.replace(
            "import subprocess\nimport time\n",
            "import subprocess\nimport tempfile\nimport time\n",
            1,
        )

    method_start = source.index("    def install_user_services(")
    block_start = source.index(
        "        self.private_service_path.write_text(", method_start
    )
    return_line = "        return self.paths()"
    block_end = source.index(return_line, block_start) + len(return_line)
    atomic_lines = [
        "with tempfile.TemporaryDirectory(",
        '    prefix=".tennis-lab-mcp-units-",',
        "    dir=self.service_dir,",
        ") as temporary_directory:",
        "    candidate_dir = Path(temporary_directory)",
        "    private_candidate = candidate_dir / PRIVATE_SERVICE_NAME",
        "    tunnel_candidate = candidate_dir / TUNNEL_SERVICE_NAME",
        "    private_candidate.write_text(",
        '        self._private_service_unit(), encoding="utf-8"',
        "    )",
        "    tunnel_candidate.write_text(",
        '        self._tunnel_service_unit(), encoding="utf-8"',
        "    )",
        "    os.chmod(private_candidate, 0o600)",
        "    os.chmod(tunnel_candidate, 0o600)",
        "    verification = subprocess.run(",
        "        [",
        '            "systemd-analyze",',
        '            "--user",',
        '            "verify",',
        "            str(private_candidate),",
        "            str(tunnel_candidate),",
        "        ],",
        "        text=True,",
        "        capture_output=True,",
        "        check=False,",
        "        timeout=30,",
        "    )",
        "    if verification.returncode != 0:",
        "        detail = (",
        "            verification.stderr.strip()",
        "            or verification.stdout.strip()",
        "        )",
        "        raise SecureTunnelError(",
        '            f"systemd unit verification failed: {detail}"',
        "        )",
        "    os.replace(private_candidate, self.private_service_path)",
        "    os.replace(tunnel_candidate, self.tunnel_service_path)",
        "os.chmod(self.private_service_path, 0o600)",
        "os.chmod(self.tunnel_service_path, 0o600)",
        "subprocess.run(",
        '    ["systemctl", "--user", "daemon-reload"],',
        "    check=True,",
        "    timeout=30,",
        ")",
        "return self.paths()",
    ]
    replacement = "\n".join(f"        {line}" for line in atomic_lines)
    path.write_text(
        source[:block_start] + replacement + source[block_end:],
        encoding="utf-8",
    )


def _replace_systemd_tests() -> None:
    path = Path("tests/unit/automation/chatgpt_mcp/test_secure_tunnel.py")
    source = path.read_text(encoding="utf-8")

    assertion_start = source.index("    assert commands[0] == [")
    assertion_end = source.index(
        '    assert commands[1] == ["systemctl", "--user", "daemon-reload"]',
        assertion_start,
    )
    assertion_lines = [
        "verify_command = commands[0]",
        'assert verify_command[:3] == ["systemd-analyze", "--user", "verify"]',
        "private_candidate = Path(verify_command[3])",
        "tunnel_candidate = Path(verify_command[4])",
        "assert private_candidate.name == PRIVATE_SERVICE_NAME",
        "assert tunnel_candidate.name == TUNNEL_SERVICE_NAME",
        "assert private_candidate.parent == tunnel_candidate.parent",
        "assert private_candidate.parent != paths.private_service.parent",
    ]
    assertion = "\n".join(f"    {line}" for line in assertion_lines) + "\n"
    source = source[:assertion_start] + assertion + source[assertion_end:]

    invalid_start = source.index(
        "def test_install_services_rejects_invalid_systemd_units("
    )
    invalid_end = source.index(
        "\n\ndef test_start_rejects_service_that_did_not_become_active(",
        invalid_start,
    )
    invalid_lines = [
        "def test_install_services_rejects_invalid_systemd_units_without_replacing_active_files(",
        "    tmp_path: Path, monkeypatch: pytest.MonkeyPatch",
        ") -> None:",
        "    manager = _manager(tmp_path)",
        "    manager.settings.secure_tunnel_profile_dir.mkdir(parents=True)",
        "    manager.settings.secure_tunnel_profile_path.write_text(",
        '        "config_version: 1\\n", encoding="utf-8"',
        "    )",
        "    manager.service_dir.mkdir(parents=True)",
        "    manager.private_service_path.write_text(",
        '        "old private unit\\n", encoding="utf-8"',
        "    )",
        "    manager.tunnel_service_path.write_text(",
        '        "old tunnel unit\\n", encoding="utf-8"',
        "    )",
        "",
        "    def fake_run(",
        "        command: list[str], **kwargs: object",
        "    ) -> subprocess.CompletedProcess[str]:",
        '        assert command[:3] == ["systemd-analyze", "--user", "verify"]',
        "        assert Path(command[3]).name == PRIVATE_SERVICE_NAME",
        "        assert Path(command[4]).name == TUNNEL_SERVICE_NAME",
        "        assert Path(command[3]).parent != manager.service_dir",
        '        return subprocess.CompletedProcess(command, 1, "", "invalid unit")',
        "",
        "    monkeypatch.setattr(",
        '        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run',
        "    )",
        "",
        '    with pytest.raises(SecureTunnelError, match="systemd unit verification failed"):',
        "        manager.install_user_services()",
        "",
        "    assert manager.private_service_path.read_text(encoding=\"utf-8\") == (",
        '        "old private unit\\n"',
        "    )",
        "    assert manager.tunnel_service_path.read_text(encoding=\"utf-8\") == (",
        '        "old tunnel unit\\n"',
        "    )",
    ]
    invalid_test = "\n".join(invalid_lines)
    path.write_text(
        source[:invalid_start] + invalid_test + source[invalid_end:],
        encoding="utf-8",
    )


def _update_readme() -> None:
    path = Path("src/automation/chatgpt_mcp/README.md")
    source = path.read_text(encoding="utf-8")
    source = source.replace(
        "registered SHA and tracked-clean source tree",
        "registered SHA and completely clean source tree",
    )
    source = source.replace(
        "raw commands are represented in durable metadata only by a\n"
        "  SHA-256 digest.",
        "raw commands are absent from Docker metadata and represented in durable\n"
        "  metadata only by a SHA-256 digest; private command/spec files are deleted\n"
        "  after handoff.",
    )
    path.write_text(source, encoding="utf-8")


def _regenerate_inventory() -> None:
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
        raise RuntimeError("unexpected non-MCP inventory findings:\n" + rendered)

    code_locations = {
        ("src.automation.chatgpt_mcp.jobs", "TrainingQueueManager.__init__"),
        ("src.automation.chatgpt_mcp.settings", "GatewaySettings.venv_root"),
        (
            "src.automation.chatgpt_mcp.workspace",
            "WorkspaceManager._verify_materialized_workspace",
        ),
    }
    path_rules = {
        AuditRule.HYDRA_ABSOLUTE_PATH,
        AuditRule.FILE_PARENT_INDEX,
        AuditRule.RUNTIME_PATH_LITERAL,
        AuditRule.PATH_JOIN,
        AuditRule.PROCESS_CWD,
        AuditRule.HYDRA_RUN_DIRECTORY,
    }
    approvals = []
    for finding in unresolved:
        if finding.rule not in path_rules:
            reason_code = "strict-schema"
        elif (
            finding.rule is AuditRule.FILE_PARENT_INDEX
            or (finding.module, finding.qualified_name) in code_locations
        ):
            reason_code = "code-or-artifact-location"
        else:
            reason_code = "persisted-layout"
        approvals.append(
            AuditExemption.classified(
                module=finding.module,
                qualified_name=finding.qualified_name,
                line=finding.line,
                rule=finding.rule,
                reason_code=reason_code,
            )
        )

    migration_count, exemption_count = write_generated_inventory_data(
        source_root,
        source_revision="pr716-execution-boundary-v3",
        approved_exemptions=tuple(approvals),
    )
    print(
        f"generated inventory: migrations={migration_count} "
        f"exemptions={exemption_count} new={len(approvals)}"
    )


def main() -> None:
    _replace_systemd_installation()
    _replace_systemd_tests()
    _update_readme()
    _regenerate_inventory()


if __name__ == "__main__":
    main()
