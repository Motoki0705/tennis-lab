"""Command-level tests for the canonical configuration audit entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_audit_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/audit_configuration.py", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("arguments", [("--help",), ("src", "--help")])
def test_root_configuration_audit_script_is_the_operational_entrypoint(
    arguments: tuple[str, ...],
) -> None:
    result = _run_audit_script(*arguments)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "source_root" in result.stdout
    assert "--show-discovered-boundaries" in result.stdout
    assert "--show-contracts" in result.stdout
    assert "--write-generated-data" not in result.stdout
    assert "--regenerate-ledger" not in result.stdout


@pytest.mark.parametrize("option", ["--write-generated-data", "--regenerate-ledger"])
def test_removed_generated_inventory_option_is_rejected(option: str) -> None:
    result = _run_audit_script("src", option)

    assert result.returncode == 2
    assert f"unrecognized arguments: {option}" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [(("--help",), 0), (("src", "--write-generated-data"), 2)],
)
def test_argument_handling_never_imports_project_code(
    arguments: tuple[str, ...], exit_code: int
) -> None:
    # -S removes installed dependencies; the audit hook also rejects stdlib-only
    # src imports, so a cheap project import cannot hide the ordering regression.
    script = """
import runpy, sys
def reject_project_import(event, arguments):
    if event == "import" and arguments[0].split(".")[0] == "src":
        raise AssertionError("project imported before argument validation")
sys.addaudithook(reject_project_import)
sys.argv = ["scripts/audit_configuration.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, *arguments],
        cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    if exit_code == 0:
        assert "source_root" in result.stdout
    else:
        assert "unrecognized arguments: --write-generated-data" in result.stderr
