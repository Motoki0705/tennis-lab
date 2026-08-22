"""Command-level regression tests for Colab role-based path overrides."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.base.configuration import (
    BaseDataConfig,
    ChunkDataConfig,
    TrainingRuntimeConfig,
)
from src.tasks.blcs.configuration import (
    parse_generation_run,
    validate_training_boundary,
)
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig

ROOT = Path(__file__).parents[3]
COLAB_ROOT = ROOT / "scripts/colab"
PATH_CONTRACT = COLAB_ROOT / "setup/path_contract.sh"
PREPARE_GENERATED = COLAB_ROOT / "setup/prepare_generated_dataset.sh"
COURT_SCRIPT = (
    COLAB_ROOT
    / "train/2026-08-22/train_court_synthetic_v2_kp_dinov3_dpt.sh"
)


@dataclass(frozen=True)
class TrainingCase:
    task: str
    script: Path
    module: str
    default_config_name: str


TRAINING_CASES = (
    TrainingCase(
        task="blcs",
        script=COLAB_ROOT / "train/2026-07-02/train_blcs_broadcast.sh",
        module="src.tasks.blcs.scripts.train",
        default_config_name="train",
    ),
    TrainingCase(
        task="plcs",
        script=COLAB_ROOT / "train/2026-07-02/train_plcs_broadcast.sh",
        module="src.tasks.plcs.scripts.train",
        default_config_name="train",
    ),
    TrainingCase(
        task="blcs",
        script=COLAB_ROOT / "train/2026-08-22/train_blcs_track_query_base.sh",
        module="src.tasks.blcs.scripts.train",
        default_config_name="train",
    ),
    TrainingCase(
        task="plcs",
        script=COLAB_ROOT / "train/2026-08-22/train_plcs_track_query_base.sh",
        module="src.tasks.plcs.scripts.train",
        default_config_name="train",
    ),
)


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


def _command_config(args: list[str], default_name: str) -> tuple[str, list[str]]:
    assert len(args) >= 2
    assert args[0] == "-m"
    config_name = default_name
    overrides: list[str] = []
    index = 2
    while index < len(args):
        if args[index] == "--config-name":
            config_name = args[index + 1]
            index += 2
        else:
            overrides.append(args[index])
            index += 1
    return config_name, overrides


def _override_mapping(overrides: list[str]) -> dict[str, str]:
    return {
        key.removeprefix("+"): value
        for override in overrides
        for key, value in [override.split("=", maxsplit=1)]
    }


def _fake_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    setup_dir = repo_root / "scripts/colab/setup"
    setup_dir.mkdir(parents=True)
    shutil.copy2(PATH_CONTRACT, setup_dir / PATH_CONTRACT.name)
    shutil.copy2(PREPARE_GENERATED, setup_dir / PREPARE_GENERATED.name)
    (setup_dir / "install_deps.sh").write_text(
        "install_colab_dependencies() { :; }\n",
        encoding="utf-8",
    )
    (setup_dir / "prepare_archive_dataset.sh").write_text(
        "prepare_archive_dataset() { :; }\n",
        encoding="utf-8",
    )
    (repo_root / "data/smplx/smplh").mkdir(parents=True)
    (repo_root / "data/ACCAD").mkdir(parents=True)
    return repo_root


@pytest.mark.parametrize(
    "script",
    (
        PATH_CONTRACT,
        PREPARE_GENERATED,
        COURT_SCRIPT,
        *(case.script for case in TRAINING_CASES),
    ),
)
def test_colab_path_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


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
    command = "source \"$1\"; shift; prepare_generated_dataset \"$@\""

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
    overrides = args[2:]
    override_map = _override_mapping(overrides)
    assert override_map["paths.data_root"] == str(data_root)
    assert override_map["run.output_dir"] == dataset_dir
    assert not Path(override_map["run.output_dir"]).is_absolute()

    config_dir = ROOT / f"src/tasks/{task}/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="generate_dataset", overrides=overrides)
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
    command = "source \"$1\"; shift; prepare_generated_dataset \"$@\""
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


@pytest.mark.parametrize("case", TRAINING_CASES, ids=lambda case: case.script.stem)
def test_train_script_emits_hydra_valid_role_paths(
    tmp_path: Path,
    case: TrainingCase,
) -> None:
    repo_root = _fake_repo_root(tmp_path)
    bin_dir, capture_path = _write_fake_python(tmp_path)
    data_root = tmp_path / "roots/data"
    artifact_root = tmp_path / "roots/artifacts"
    output_root = tmp_path / "roots/outputs"
    checkpoint_root = tmp_path / "roots/checkpoints"
    dataset_dir = f"{case.task}/captured-scenes"
    chunks_dir = f"{case.task}/captured-chunks"
    output_dir = f"{case.task}-captured-run"
    dataset_path = data_root / dataset_dir
    dataset_path.mkdir(parents=True)
    (dataset_path / "meta.json").write_text("{}\n", encoding="utf-8")
    env = {
        **os.environ,
        "ARGS_CAPTURE": str(capture_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "REPO_ROOT": str(repo_root),
        "DATA_ROOT": str(data_root),
        "DATASET_DIR": dataset_dir,
        "ARTIFACT_ROOT": str(artifact_root),
        "CHUNKS_DIR": chunks_dir,
        "OUTPUT_ROOT": str(output_root),
        "OUTPUT_DIR": output_dir,
        "CHECKPOINT_ROOT": str(checkpoint_root),
    }

    result = subprocess.run(
        ["bash", str(case.script)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    commands = _read_commands(capture_path)
    assert len(commands) == 1
    args = commands[0]
    assert args[:2] == ["-m", case.module]
    config_name, overrides = _command_config(args, case.default_config_name)
    override_map = _override_mapping(overrides)
    expected = {
        "paths.data_root": str(data_root),
        "paths.artifact_root": str(artifact_root),
        "paths.output_root": str(output_root),
        "paths.checkpoint_root": str(checkpoint_root),
        "data.scene_dir": dataset_dir,
        "data.chunk.chunks_dir": chunks_dir,
        "run.output_dir": output_dir,
    }
    assert {key: override_map[key] for key in expected} == expected
    for key in ("data.scene_dir", "data.chunk.chunks_dir", "run.output_dir"):
        assert not Path(override_map[key]).is_absolute()

    config_dir = ROOT / f"src/tasks/{case.task}/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name=config_name, overrides=overrides)
    if case.task == "blcs":
        validate_training_boundary(config)
        shared = TrainingRuntimeConfig.from_config(config, repository_root=ROOT)
        resolver = shared.resolver
    else:
        plcs_runtime = PLCSTrainingConfig.from_config(config)
        shared = plcs_runtime.shared
        resolver = plcs_runtime.paths.resolver
    data = BaseDataConfig.from_validated_task_mapping(config.data, resolver=resolver)
    chunk = ChunkDataConfig.from_validated_task_mapping(config.data, resolver=resolver)
    assert data.scene_dir == data_root / dataset_dir
    assert chunk.chunks_dir == artifact_root / chunks_dir
    assert shared.run.output_dir == output_root / output_dir


def test_court_train_script_emits_hydra_valid_role_paths_and_resume(
    tmp_path: Path,
) -> None:
    repo_root = _fake_repo_root(tmp_path)
    bin_dir, capture_path = _write_fake_python(tmp_path)
    data_root = tmp_path / "roots/data"
    output_root = tmp_path / "roots/outputs"
    checkpoint_root = output_root
    output_dir = "court_detection/captured-synthetic-v2"
    dataset_root = (
        data_root / "synthetic_data_generation/scenes/B00/datasets/court"
    )
    dataset_root.mkdir(parents=True)
    (dataset_root / "dataset.json").write_text(
        '{"schema": "canonical_court_dataset_v2"}\n',
        encoding="utf-8",
    )
    (repo_root / "third_party/dinov3/dinov3").mkdir(parents=True)
    checkpoint = (
        output_root / output_dir / "logs/version_0/checkpoints/last.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    backbone = (
        repo_root
        / "third_party/dinov3/checkpoints"
        / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    backbone.parent.mkdir(parents=True)
    backbone.touch()
    env = {
        **os.environ,
        "ARGS_CAPTURE": str(capture_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "REPO_ROOT": str(repo_root),
        "DATA_ROOT": str(data_root),
        "OUTPUT_ROOT": str(output_root),
        "OUTPUT_DIR": output_dir,
        "CHECKPOINT_ROOT": str(checkpoint_root),
    }

    result = subprocess.run(
        ["bash", str(COURT_SCRIPT)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    commands = _read_commands(capture_path)
    assert len(commands) == 1
    args = commands[0]
    assert args[:2] == ["-m", "src.tasks.court_detection.scripts.train"]
    _config_name, overrides = _command_config(args, "train")
    override_map = _override_mapping(overrides)
    expected = {
        "paths.data_root": str(data_root),
        "paths.output_root": str(output_root),
        "paths.checkpoint_root": str(checkpoint_root),
        "data.source.workspace_root": "synthetic_data_generation/scenes",
        "run.output_dir": output_dir,
        "run.resume": f"{output_dir}/logs/version_0/checkpoints/last.ckpt",
    }
    assert {key: override_map[key] for key in expected} == expected
    for key in ("data.source.workspace_root", "run.output_dir", "run.resume"):
        assert not Path(override_map[key]).is_absolute()

    config_dir = ROOT / "src/tasks/court_detection/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train", overrides=overrides)
    runtime = CourtTrainingConfig.from_config(config)
    assert runtime.data.source.workspace_root == (
        data_root / "synthetic_data_generation/scenes"
    )
    assert runtime.shared.run.output_dir == output_root / output_dir
    assert runtime.shared.run.resume == checkpoint
