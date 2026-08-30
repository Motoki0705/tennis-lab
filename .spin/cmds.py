"""Project-specific developer commands exposed through :mod:`spin`."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
NHT_ROOT_RELATIVE = Path("third_party/nht")
NHT_TRAINER_VENV_RELATIVE = NHT_ROOT_RELATIVE / ".trainer-venv"
NHT_TRAINER_PYTHON_RELATIVE = NHT_TRAINER_VENV_RELATIVE / "bin/python"
NHT_TRAINER_REQUIREMENTS_RELATIVE = NHT_ROOT_RELATIVE / "gsplat/examples/requirements.txt"
NHT_TRAINER_RELATIVE = NHT_ROOT_RELATIVE / "gsplat/examples/simple_trainer_nht.py"
NHT_ADAPTER_RELATIVE = NHT_ROOT_RELATIVE / "nht_pipeline/nht_adapter.py"
NHT_PUBLIC_COMMANDS = ("nht-reconstruct", "nht-render")
NHT_TOOL_ENVIRONMENT = "nht"
NHT_PYTHON_VERSION = "3.11"
NHT_TORCH_VERSION = "2.9.1"
NHT_TORCHVISION_VERSION = "0.24.1"
NHT_TORCH_BACKEND = "cu130"
NHT_TINYCUDANN_REQUIREMENT = (
    "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/"
    "@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
)
DEFAULT_BASE = "origin/main"
DEFAULT_LINT_PATHS = ("src", "tests", ".spin")
DEFAULT_TYPECHECK_PATHS = ("src", "tests", ".spin/cmds.py")
CI_MARKER_EXPRESSION = "not local_data and not cuda"
CI_LANES = ("remainder", "long-tail", "scene-pipeline")
CI_LONG_TAIL_TEST_FILES = frozenset(
    {
        "tests/e2e/development/test_configuration_audit.py",
        "tests/unit/synthetic_data_generation/alignment/test_evidence_source.py",
        "tests/unit/tasks/plcs/test_configuration_contracts.py",
        "tests/unit/utils/configuration/test_audit.py",
        "tests/unit/utils/configuration/test_discovery.py",
        "tests/unit/utils/configuration/test_inventory.py",
    }
)
CI_SCENE_PIPELINE_TEST_FILES = frozenset(
    {"tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py"}
)
CI_SPECIALIZED_TEST_FILES = (
    CI_LONG_TAIL_TEST_FILES | CI_SCENE_PIPELINE_TEST_FILES
)
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    click.secho(f"$ {shlex.join(command)}", dim=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        env=dict(environment) if environment is not None else None,
    )
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
            if Path(name).suffix in PYTHON_SUFFIXES and (REPO_ROOT / name).is_file()
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


def _discover_ci_test_files(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return every pytest file using repository-relative POSIX paths."""
    tests_root = repo_root / "tests"
    if not tests_root.is_dir():
        raise FileNotFoundError(f"Tests directory is unavailable: {tests_root}")

    test_files = tuple(
        sorted(
            path.relative_to(repo_root).as_posix()
            for path in tests_root.rglob("*.py")
            if path.name.startswith("test_") or path.name.endswith("_test.py")
        )
    )
    if not test_files:
        raise RuntimeError(f"No pytest files were found under {tests_root}")
    return test_files


def _select_ci_test_files(
    lane: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    """Return exactly the test files assigned to one GitHub Actions lane."""
    if lane not in CI_LANES:
        choices = ", ".join(CI_LANES)
        raise ValueError(f"Unknown CI test lane {lane!r}; expected one of: {choices}")

    all_files = frozenset(_discover_ci_test_files(repo_root))
    missing = sorted(CI_SPECIALIZED_TEST_FILES - all_files)
    if missing:
        rendered = ", ".join(missing)
        raise FileNotFoundError(
            f"Configured specialized test files are unavailable: {rendered}"
        )

    if lane == "long-tail":
        selected = CI_LONG_TAIL_TEST_FILES
    elif lane == "scene-pipeline":
        selected = CI_SCENE_PIPELINE_TEST_FILES
    else:
        selected = all_files - CI_SPECIALIZED_TEST_FILES

    if not selected:
        raise RuntimeError(f"CI test lane {lane!r} is empty")
    return tuple(sorted(selected))


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


def _uv_tool_bin_directory() -> Path | None:
    result = subprocess.run(
        ["uv", "tool", "dir", "--bin"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    rendered = result.stdout.strip()
    return Path(rendered) if rendered else None


def _uv_tool_directory() -> Path | None:
    result = subprocess.run(
        ["uv", "tool", "dir"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    rendered = result.stdout.strip()
    return Path(rendered) if rendered else None


def _nht_cuda_build_environment() -> Mapping[str, str]:
    """Expose WSL's CUDA driver import library only while extensions build."""
    environment = dict(os.environ)
    wsl_driver_directory = Path("/usr/lib/wsl/lib")
    if (wsl_driver_directory / "libcuda.so").is_file():
        current = environment.get("LIBRARY_PATH")
        entries = [str(wsl_driver_directory)]
        if current:
            entries.append(current)
        environment["LIBRARY_PATH"] = os.pathsep.join(entries)
    return environment


@click.command("setup-nht")
@click.option(
    "--with-sfm-learned",
    is_flag=True,
    help="Install the optional HLOC/ALIKED/LightGlue retry backend.",
)
def setup_nht(with_sfm_learned: bool) -> None:
    """Install isolated NHT public CLIs and its dedicated trainer runtime."""
    relative_root = NHT_ROOT_RELATIVE.as_posix()
    _run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--checkout",
            relative_root,
        ]
    )
    extras = ["aov"]
    if with_sfm_learned:
        extras.append("sfm-learned")
    package = f"{relative_root}[{','.join(extras)}]"
    _run(
        [
            "uv",
            "tool",
            "install",
            "--force",
            "--python",
            NHT_PYTHON_VERSION,
            "--editable",
            "--with-editable",
            f"{relative_root}/gsplat",
            "--with",
            f"torch=={NHT_TORCH_VERSION}",
            "--with",
            f"torchvision=={NHT_TORCHVISION_VERSION}",
            "--torch-backend",
            NHT_TORCH_BACKEND,
            package,
        ]
    )
    tool_directory = _uv_tool_directory()
    if tool_directory is None:
        raise click.ClickException(
            "NHT was installed, but `uv tool dir` did not return a directory."
        )
    tool_python = tool_directory / NHT_TOOL_ENVIRONMENT / "bin/python"
    cuda_build_environment = _nht_cuda_build_environment()
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tool_python),
            "setuptools<81",
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tool_python),
            "--no-build-isolation",
            "--no-cache",
            NHT_TINYCUDANN_REQUIREMENT,
        ],
        environment=cuda_build_environment,
    )
    _run(
        [
            "uv",
            "run",
            "--python",
            str(tool_python),
            "--no-project",
            "--",
            "python",
            "-c",
            "from gsplat.nht.deferred_shader import DeferredShaderModule",
        ]
    )
    trainer_python = NHT_TRAINER_PYTHON_RELATIVE.as_posix()
    _run(
        [
            "uv",
            "venv",
            "--clear",
            "--python",
            NHT_PYTHON_VERSION,
            NHT_TRAINER_VENV_RELATIVE.as_posix(),
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            trainer_python,
            "setuptools<81",
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            trainer_python,
            "--torch-backend",
            NHT_TORCH_BACKEND,
            f"torch=={NHT_TORCH_VERSION}",
            f"torchvision=={NHT_TORCHVISION_VERSION}",
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            trainer_python,
            "--no-build-isolation",
            "--no-cache",
            "--requirements",
            NHT_TRAINER_REQUIREMENTS_RELATIVE.as_posix(),
        ],
        environment=cuda_build_environment,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            trainer_python,
            "--no-build-isolation",
            "--no-cache",
            "--editable",
            f"{relative_root}/gsplat",
        ],
        environment=cuda_build_environment,
    )
    _run(
        [
            "uv",
            "run",
            "--python",
            trainer_python,
            "--no-project",
            NHT_ADAPTER_RELATIVE.as_posix(),
            "probe",
            "--trainer",
            NHT_TRAINER_RELATIVE.as_posix(),
        ]
    )

    tool_bin = _uv_tool_bin_directory()
    if tool_bin is None:
        raise click.ClickException(
            "NHT was installed, but `uv tool dir --bin` did not return a directory."
        )
    installed_commands = {
        command: tool_bin / command for command in NHT_PUBLIC_COMMANDS
    }
    missing_installed = [
        command
        for command, path in installed_commands.items()
        if not path.is_file() or not os.access(path, os.X_OK)
    ]
    if missing_installed:
        raise click.ClickException(
            "NHT installation did not publish its required public commands in "
            f"{tool_bin}: {', '.join(missing_installed)}."
        )

    missing_from_path = [
        command for command in NHT_PUBLIC_COMMANDS if shutil.which(command) is None
    ]
    if missing_from_path:
        raise click.ClickException(
            "NHT was installed, but its public commands are not on PATH: "
            f"{', '.join(missing_from_path)}. Add {tool_bin} to PATH (or run `uv tool "
            "update-shell` and restart the shell)."
        )

    mismatched = []
    for command, installed in installed_commands.items():
        discovered = shutil.which(command)
        if discovered is None:
            raise RuntimeError("Validated NHT command disappeared from PATH.")
        if Path(discovered).resolve() != installed.resolve():
            mismatched.append(f"{command}={discovered}")
    if mismatched:
        raise click.ClickException(
            "PATH resolves NHT commands outside the installed uv tool environment: "
            f"{', '.join(mismatched)} (expected directory: {tool_bin})."
        )

    click.echo("NHT public CLI is ready:")
    for command, installed in installed_commands.items():
        click.echo(f"  {command}: {installed}")
    click.echo(f"NHT trainer runtime is ready: {REPO_ROOT / NHT_TRAINER_VENV_RELATIVE}")


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
@click.option(
    "--lane",
    type=click.Choice(CI_LANES),
    help="Run one GitHub Actions test lane instead of the complete suite.",
)
@click.option(
    "--list-tests",
    is_flag=True,
    help="Print selected test files without running checks.",
)
def ci(lane: str | None, list_tests: bool) -> None:
    """Run the repository-wide checks used by GitHub Actions."""
    try:
        selected = (
            _discover_ci_test_files()
            if lane is None
            else _select_ci_test_files(lane)
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if list_tests:
        for test_file in selected:
            click.echo(test_file)
        return

    if lane in {None, "remainder"}:
        _run_lint(DEFAULT_LINT_PATHS)

    parallel_args = (
        ("-n", "0")
        if lane == "scene-pipeline"
        else ("-n", "auto", "--dist=worksteal")
    )
    _run(
        _pytest_command(
            include_environmental=False,
            coverage=False,
            serial=False,
            extra_args=(
                *selected,
                "-q",
                "--no-cov",
                *parallel_args,
                "--durations=25",
            ),
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
            _command_check(
                "nht-reconstruct",
                required=False,
                purpose="synthetic scene reconstruction; run `spin setup-nht`",
            ),
            _command_check(
                "nht-render",
                required=False,
                purpose="synthetic dataset rendering; run `spin setup-nht`",
            ),
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
