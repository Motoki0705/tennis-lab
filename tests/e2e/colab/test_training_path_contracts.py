"""Command-level regression tests for Colab role-based path overrides.

The historical Colab training entrypoints are retired.  These tests cover the
remaining setup helpers that still form part of the supported path contract.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.configuration import parse_generation_run
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig

ROOT = Path(__file__).parents[3]
COLAB_ROOT = ROOT / "scripts/colab"
PATH_CONTRACT = COLAB_ROOT / "setup/path_contract.sh"
PREPARE_GENERATED = COLAB_ROOT / "setup/prepare_generated_dataset.sh"
PREPARE_ARCHIVE = COLAB_ROOT / "setup/prepare_archive_dataset.sh"
INSTALL_CUDA_OPS = COLAB_ROOT / "setup/install_cuda_ops.sh"


def _write_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    capture_path = tmp_path / "python-commands.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_path = bin_dir / "python"
    python_path.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    printf '%s\n' 'canonical_court_dataset_v2'
    exit 0
fi
{
    printf '%s\n' '__COMMAND__'
    printf '%s\n' "$@"
} >> "${ARGS_CAPTURE:?}"
""",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return bin_dir, capture_path


def _read_commands(capture_path: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] | None = None
    for line in capture_path.read_text(encoding="utf-8").splitlines():
        if line == "__COMMAND__":
            current = []
            commands.append(current)
        elif current is not None:
            current.append(line)
    return commands


def _override_mapping(overrides: list[str]) -> dict[str, str]:
    return {
        key.removeprefix("+"): value
        for override in overrides
        for key, value in [override.split("=", maxsplit=1)]
    }


@pytest.mark.parametrize(
    "script",
    (PATH_CONTRACT, PREPARE_GENERATED, PREPARE_ARCHIVE, INSTALL_CUDA_OPS),
)
def test_colab_setup_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_smplh_archive_staging_preserves_both_formats_and_originals(tmp_path: Path) -> None:
    data, repo = tmp_path / 'data', tmp_path / 'repo'
    sources = {'smplh/male/model.npz': b'npz-body-model', 'smplx/smplh/SMPLH_MALE.pkl': b'pkl-body-model'}
    for name, value in sources.items():
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    subprocess.run(['bash', '-c', 'source "$1"; _stage_smplh_models "$2" "$3"',
        'stage-smplh', str(PREPARE_ARCHIVE), str(data), str(repo)], check=True, capture_output=True)
    assert (repo / 'ckpt/body_models/smplh/male/model.npz').read_bytes() == sources['smplh/male/model.npz']
    assert (repo / 'ckpt/body_models/smplh/SMPLH_MALE.pkl').read_bytes() == sources['smplx/smplh/SMPLH_MALE.pkl']
    for name, value in sources.items():
        assert (data / name).read_bytes() == value


def test_smplh_archive_staging_rejects_incomplete_assets(tmp_path: Path) -> None:
    data, repo = tmp_path / 'data', tmp_path / 'repo'
    (data / 'smplh').mkdir(parents=True)
    result = subprocess.run(['bash', '-c', 'source "$1"; _stage_smplh_models "$2" "$3"',
        'stage-smplh', str(PREPARE_ARCHIVE), str(data), str(repo)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'both SMPL-H NPZ and PKL' in result.stderr
    assert not (repo / 'ckpt/body_models').exists()


@pytest.mark.parametrize(
    ("task", "module"),
    (
        ("blcs", "src.tasks.blcs.scripts.generate_dataset"),
        ("plcs", "src.tasks.plcs.scripts.generate_dataset"),
    ),
)
def test_generated_dataset_helper_emits_root_and_relative_child(
    tmp_path: Path,
    task: str,
    module: str,
) -> None:
    bin_dir, capture_path = _write_fake_python(tmp_path)
    data_root = tmp_path / "absolute-data-root"
    dataset_dir = f"{task}/captured-scenes"
    env = {
        **os.environ,
        "ARGS_CAPTURE": str(capture_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
    }
    command = 'source "$1"; shift; prepare_generated_dataset "$@"'

    subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(PREPARE_GENERATED),
            task,
            str(ROOT),
            str(data_root),
            dataset_dir,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    commands = _read_commands(capture_path)
    assert len(commands) == 1
    args = commands[0]
    assert args[:2] == ["-m", module]
    override_map = _override_mapping(args[2:])
    assert override_map["paths.data_root"] == str(data_root)
    assert override_map["run.output_dir"] == dataset_dir
    assert not Path(override_map["run.output_dir"]).is_absolute()

    config_dir = ROOT / f"src/tasks/{task}/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="generate_dataset", overrides=args[2:])
    if task == "blcs":
        runtime, _resolver = parse_generation_run(config)
    else:
        runtime = PLCSGenerationConfig.from_config(config)
    assert runtime.output_dir == data_root / dataset_dir


def test_generated_dataset_helper_rejects_absolute_child_before_python(
    tmp_path: Path,
) -> None:
    bin_dir, capture_path = _write_fake_python(tmp_path)
    env = {
        **os.environ,
        "ARGS_CAPTURE": str(capture_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
    }
    command = 'source "$1"; shift; prepare_generated_dataset "$@"'
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(PREPARE_GENERATED),
            "blcs",
            str(ROOT),
            str(tmp_path / "data-root"),
            str(tmp_path / "absolute-child"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "must be role-relative, not absolute" in result.stderr
    assert not capture_path.exists()
