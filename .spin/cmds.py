"""Project-specific developer commands exposed through :mod:`spin`."""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "origin/main"
DEFAULT_LINT_PATHS = ("src", "tests", ".spin")
DEFAULT_TYPECHECK_PATHS = ("src", "tests", ".spin/cmds.py")
CI_MARKER_EXPRESSION = "not local_data and not cuda"
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})


def _run(command: Sequence[str]) -> None:
    click.secho(f"$ {shlex.join(command)}", dim=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise click.exceptions.Exit(result.returncode)


def _git_output(args: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise click.ClickException(detail or f"git {' '.join(args)} failed")
    return result.stdout


def _changed_python_files(base: str) -> tuple[str, ...]:
    base_check = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if base_check.returncode != 0:
        raise click.ClickException(
            f"Git base {base!r} was not found. Fetch it or pass --base <ref>."
        )

    scope = list(DEFAULT_LINT_PATHS)
    changed = _git_output(
        ["diff", "--name-only", "--diff-filter=ACMR", "-z", base, "--", *scope]
    )
    untracked = _git_output(
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *scope]
    )
    names = {
        name
        for name in (changed + untracked).decode(errors="surrogateescape").split("\0")
        if name
    }
    return tuple(
        sorted(
            name
            for name in names
            if Path(name).suffix in PYTHON_SUFFIXES
            and (REPO_ROOT / name).is_file()
        )
    )


def _run_lint(paths: Sequence[str], *, fix: bool = False) -> None:
    command = [sys.executable, "-m", "ruff", "check"]
    if fix:
        command.append("--fix")
    command.extend(paths)
    _run(command)


def _run_typecheck(paths: Sequence[str]) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--follow-imports=skip",
            *paths,
        ]
    )


def _pytest_command(
    *,
    include_environmental: bool,
    coverage: bool,
    serial: bool,
    extra_args: Sequence[str],
) -> list[str]:
    command = [sys.executable, "-m", "pytest"]
    if not include_environmental:
        command.extend(["-m", CI_MARKER_EXPRESSION])
    if coverage:
        command.extend(["--cov=src", "--cov-report=term-missing"])
    if serial:
        command.extend(["-n", "0"])
    command.extend(extra_args)
    return command


@click.command()
@click.option(
    "--no-hooks",
    is_flag=True,
    help="Synchronize dependencies without installing the pre-commit hook.",
)
def setup(no_hooks: bool) -> None:
    """Synchronize the locked development environment and Git hooks."""
    _run(["uv", "sync", "--locked"])
    if not no_hooks:
        _run([sys.executable, "-m", "pre_commit", "install"])


@click.command()
@click.option("--fix", is_flag=True, help="Apply Ruff's safe automatic fixes.")
@click.option(
    "--changed",
    is_flag=True,
    help="Check Python files changed from --base, including untracked files.",
)
@click.option(
    "--base",
    default=DEFAULT_BASE,
    show_default=True,
    help="Git revision used by --changed.",
)
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
def lint(fix: bool, changed: bool, base: str, paths: tuple[Path, ...]) -> None:
    """Run Ruff on the source, tests, and developer commands."""
    if changed and paths:
        raise click.UsageError("PATHS and --changed cannot be used together.")
    selected = (
        _changed_python_files(base)
        if changed
        else tuple(str(path) for path in paths) or DEFAULT_LINT_PATHS
    )
    if not selected:
        click.echo("No changed Python files to lint.")
        return
    _run_lint(selected, fix=fix)


@click.command()
@click.option(
    "--all",
    "all_files",
    is_flag=True,
    help="Check the complete source and test trees, including existing debt.",
)
@click.option(
    "--base",
    default=DEFAULT_BASE,
    show_default=True,
    help="Git revision used when neither PATHS nor --all is given.",
)
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
def typecheck(all_files: bool, base: str, paths: tuple[Path, ...]) -> None:
    """Run mypy on explicit paths or Python files changed from a Git base."""
    if all_files and paths:
        raise click.UsageError("PATHS and --all cannot be used together.")
    if paths:
        selected = tuple(str(path) for path in paths)
    elif all_files:
        selected = DEFAULT_TYPECHECK_PATHS
    else:
        selected = _changed_python_files(base)
    if not selected:
        click.echo("No changed Python files to type-check.")
        return
    _run_typecheck(selected)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--all",
    "include_environmental",
    is_flag=True,
    help="Include tests marked local_data or cuda.",
)
@click.option("--coverage", is_flag=True, help="Collect coverage for src/.")
@click.option(
    "--serial",
    is_flag=True,
    help="Disable the repository's default pytest-xdist parallelism.",
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def test(
    include_environmental: bool,
    coverage: bool,
    serial: bool,
    pytest_args: tuple[str, ...],
) -> None:
    """Run pytest, forwarding remaining arguments directly to pytest."""
    if not include_environmental:
        click.echo(f"Excluding pytest markers: {CI_MARKER_EXPRESSION}")
    _run(
        _pytest_command(
            include_environmental=include_environmental,
            coverage=coverage,
            serial=serial,
            extra_args=pytest_args,
        )
    )


@click.command()
def ci() -> None:
    """Run the same repository-wide checks as GitHub Actions."""
    _run_lint(DEFAULT_LINT_PATHS)
    _run(
        _pytest_command(
            include_environmental=False,
            coverage=False,
            serial=False,
            extra_args=("-q", "--no-cov"),
        )
    )


DoctorStatus = Literal["ok", "warning", "error"]


class DoctorResult:
    """One environment diagnostic and its severity."""

    __slots__ = ("detail", "label", "status")

    def __init__(self, label: str, status: DoctorStatus, detail: str) -> None:
        self.label = label
        self.status = status
        self.detail = detail


def _command_check(command: str, *, required: bool, purpose: str) -> DoctorResult:
    location = shutil.which(command)
    if location is not None:
        return DoctorResult(command, "ok", location)
    status: DoctorStatus = "error" if required else "warning"
    return DoctorResult(command, status, f"not found ({purpose})")


def _module_check(module: str) -> DoctorResult:
    if importlib.util.find_spec(module) is not None:
        return DoctorResult(f"Python module {module}", "ok", "available")
    return DoctorResult(
        f"Python module {module}",
        "error",
        "missing; run `uv sync --locked`",
    )


def _lockfile_check() -> DoctorResult:
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return DoctorResult("Lockfile", "ok", "uv.lock matches pyproject.toml")
    output = (result.stderr or result.stdout).strip().splitlines()
    detail = output[-1] if output else "uv lock --check failed"
    return DoctorResult("Lockfile", "error", detail)


def _submodule_check() -> DoctorResult:
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git submodule status failed"
        return DoctorResult("Git submodules", "error", detail)

    lines = [line for line in result.stdout.splitlines() if line]
    conflicted = [line.split(maxsplit=1)[-1] for line in lines if line.startswith("U")]
    if conflicted:
        return DoctorResult(
            "Git submodules",
            "error",
            f"conflicted: {', '.join(conflicted)}",
        )
    uninitialized = [
        line.split(maxsplit=1)[-1] for line in lines if line.startswith("-")
    ]
    mismatched = [line.split(maxsplit=1)[-1] for line in lines if line.startswith("+")]
    details: list[str] = []
    if uninitialized:
        details.append(f"uninitialized: {', '.join(uninitialized)}")
    if mismatched:
        details.append(f"different commit: {', '.join(mismatched)}")
    if details:
        return DoctorResult("Git submodules", "warning", "; ".join(details))
    return DoctorResult("Git submodules", "ok", f"{len(lines)} initialized")


def _doctor_results() -> list[DoctorResult]:
    results = [
        DoctorResult("Python", "ok", f"{sys.version.split()[0]} (requires >=3.11)")
    ]

    expected_prefix = (REPO_ROOT / ".venv").resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix == expected_prefix:
        results.append(DoctorResult("Virtual environment", "ok", str(actual_prefix)))
    else:
        results.append(
            DoctorResult(
                "Virtual environment",
                "error",
                f"using {actual_prefix}; expected {expected_prefix}",
            )
        )

    results.extend(
        [
            _command_check("git", required=True, purpose="source control"),
            _command_check("uv", required=True, purpose="dependency management"),
            _command_check("ffmpeg", required=False, purpose="video e2e tests"),
            _command_check("rclone", required=False, purpose="Google Drive workflows"),
            _command_check("nvidia-smi", required=False, purpose="local CUDA training"),
        ]
    )
    results.extend(
        _module_check(module)
        for module in ("spin", "pytest", "ruff", "mypy", "pre_commit")
    )
    if shutil.which("uv") is not None:
        results.append(_lockfile_check())
    if shutil.which("git") is not None:
        results.append(_submodule_check())
    return results


@click.command()
@click.option(
    "--strict",
    is_flag=True,
    help="Treat optional-tool and submodule warnings as failures.",
)
def doctor(strict: bool) -> None:
    """Diagnose the local development environment without changing it."""
    results = _doctor_results()
    symbols: dict[DoctorStatus, str] = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
    colors: dict[DoctorStatus, str] = {
        "ok": "green",
        "warning": "yellow",
        "error": "red",
    }
    for result in results:
        click.secho(
            f"{symbols[result.status]:>5}  {result.label}: {result.detail}",
            fg=colors[result.status],
        )

    errors = [result for result in results if result.status == "error"]
    warnings = [result for result in results if result.status == "warning"]
    if errors or (strict and warnings):
        failure_count = len(errors) + (len(warnings) if strict else 0)
        raise click.ClickException(f"{failure_count} environment check(s) failed")
