"""Command-level tests for the Colab CUDA-operation setup module."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
INSTALL_CUDA_OPS = ROOT / "scripts/colab/setup/install_cuda_ops.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _fake_runtime(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "setup.py").touch()

    capture_path = tmp_path / "commands.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    _write_executable(cuda_home / "bin/nvcc", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(bin_dir / "g++", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "git",
        "#!/usr/bin/env bash\nprintf '%s\\n' 'fake-revision'\n",
    )
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
{
    printf '%s\n' '__COMMAND__'
    printf '%s\n' "$@"
} >> "${COMMAND_CAPTURE:?}"

if [[ "${1:-}" == "-" ]]; then
    script=""
    while IFS= read -r line; do
        script+="${line}"$'\n'
    done
    if [[ "${script}" == *"torch.cuda.is_available"* && "${script}" == *"CUDA_HOME"* ]]; then
        printf '%s\n' "${FAKE_CUDA_HOME:?}"
    elif [[ "${script}" == *"device_capability"* ]]; then
        printf '%s\n' "${FAKE_BUILD_SIGNATURE:?}"
    elif [[ "${script}" == *"repository_root = sys.argv[1]"* ]]; then
        printf '%s\n' '{"strict":"build-config"}'
    else
        printf '%s\n' 'unexpected Python stdin program' >&2
        exit 90
    fi
    exit 0
fi

if [[ "${1:-}" == "-c" ]]; then
    if [[ "${FAKE_IMPORT_FAILURE:-0}" == "1" ]]; then
        exit 91
    fi
    exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    exit 0
fi

if [[ "${1:-}" == "setup.py" ]]; then
    {
        printf '%s\n' '__BUILD_ENV__'
        printf '%s\n' "${MAX_JOBS:-}"
        printf '%s\n' "${TENNIS_LAB_BUILD_CUDA_OPS:-}"
        printf '%s\n' "${TENNIS_LAB_CUDA_OPS_BUILD_TARGET:-}"
        printf '%s\n' "${TENNIS_LAB_DINO_OPS_BUILD_CONFIG:-}"
    } >> "${COMMAND_CAPTURE:?}"
    exit "${FAKE_BUILD_EXIT:-0}"
fi

printf '%s\n' 'unexpected fake python invocation' >&2
exit 92
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "COMMAND_CAPTURE": str(capture_path),
        "FAKE_CUDA_HOME": str(cuda_home),
        "FAKE_BUILD_SIGNATURE": '{"schema":1,"runtime":"fake"}',
    }
    return repo_root, capture_path, cuda_home, environment


def _run_installer(
    repo_root: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_colab_cuda_ops "$2"',
            "bash",
            str(INSTALL_CUDA_OPS),
            str(repo_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_cuda_op_is_built_once_then_reused(tmp_path: Path) -> None:
    repo_root, capture_path, _cuda_home, environment = _fake_runtime(tmp_path)
    environment["CUDA_OPS_MAX_JOBS"] = "3"

    first = _run_installer(repo_root, environment)
    second = _run_installer(repo_root, environment)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "already built for this runtime" in second.stdout
    captured = capture_path.read_text(encoding="utf-8")
    assert captured.count("setup.py\n") == 1
    assert "__BUILD_ENV__\n3\n1\ncompressed_time_local\n" in captured
    assert "build_ext\n--inplace\n--force\n" in captured
    signature_path = repo_root / ".cache/colab_cuda_ops/build_signature.json"
    assert signature_path.read_text(encoding="utf-8") == (
        environment["FAKE_BUILD_SIGNATURE"] + "\n"
    )


def test_cuda_op_build_failure_is_hard_and_leaves_no_signature(
    tmp_path: Path,
) -> None:
    repo_root, _capture_path, _cuda_home, environment = _fake_runtime(tmp_path)
    environment["FAKE_BUILD_EXIT"] = "9"

    completed = _run_installer(repo_root, environment)

    assert completed.returncode != 0
    assert "CUDA operation build failed" in completed.stderr
    assert not (repo_root / ".cache/colab_cuda_ops/build_signature.json").exists()


@pytest.mark.parametrize("value", ["", "0", "2.5", "four", " 2"])
def test_cuda_op_build_rejects_invalid_max_jobs(tmp_path: Path, value: str) -> None:
    repo_root, _capture_path, _cuda_home, environment = _fake_runtime(tmp_path)
    environment["CUDA_OPS_MAX_JOBS"] = value

    completed = _run_installer(repo_root, environment)

    assert completed.returncode == 2
    assert "must be a positive integer" in completed.stderr
