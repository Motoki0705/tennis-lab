from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.synthetic_data_generation.reconstruction import nht_subprocess
from src.synthetic_data_generation.reconstruction.contracts import (
    NHT_PIPELINE_CONFIG_SCHEMA,
    NHTPipelineConfig,
    NHTTrainingRuntime,
    ReconstructionCommandRequest,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)


def _pipeline_config(tmp_path: Path) -> NHTPipelineConfig:
    path = tmp_path / "nht-pipeline.yaml"
    path.write_text(
        f"schema: {NHT_PIPELINE_CONFIG_SCHEMA}\n",
        encoding="utf-8",
    )
    return NHTPipelineConfig.load(path.resolve())


def _request(tmp_path: Path) -> ReconstructionCommandRequest:
    scene_root = tmp_path / "scenes/B00"
    source = scene_root / "source/video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    return ReconstructionCommandRequest(
        scene_id="B00",
        input_video=source,
        workspace=scene_root / "reconstruction",
        pipeline_config=_pipeline_config(tmp_path),
    )


def _training_runtime(tmp_path: Path) -> NHTTrainingRuntime:
    python = tmp_path / "trainer/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    trainer = tmp_path / "trainer/simple_trainer_nht.py"
    trainer.write_text("# test trainer\n", encoding="utf-8")
    return NHTTrainingRuntime(
        python=python.resolve(),
        trainer=trainer.resolve(),
    )


def test_training_runtime_rejects_missing_python(tmp_path: Path) -> None:
    trainer = tmp_path / "simple_trainer_nht.py"
    trainer.write_text("# test trainer\n", encoding="utf-8")
    runtime = NHTTrainingRuntime(
        python=(tmp_path / "missing-python").resolve(),
        trainer=trainer.resolve(),
    )

    with pytest.raises(FileNotFoundError, match="training Python is unavailable"):
        runtime.validate()


def test_training_runtime_rejects_missing_trainer(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    runtime = NHTTrainingRuntime(
        python=python.resolve(),
        trainer=(tmp_path / "missing-trainer.py").resolve(),
    )

    with pytest.raises(FileNotFoundError, match="trainer is unavailable"):
        runtime.validate()


def test_request_builds_fixed_public_command(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert request.argv() == (
        "nht-reconstruct",
        "--scene-id",
        "B00",
        "--input-video",
        str(request.input_video),
        "--workspace",
        str(request.workspace),
        "--config",
        str(request.pipeline_config.path),
    )
    assert request.scene_path == request.workspace / "export/scene.json"


def test_request_rejects_noncanonical_workspace(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="fixed <scene_id>/reconstruction"):
        ReconstructionCommandRequest(
            scene_id=request.scene_id,
            input_video=request.input_video,
            workspace=tmp_path / "runs/attempt-1",
            pipeline_config=request.pipeline_config,
        )


def test_request_accepts_an_installed_absolute_public_command(tmp_path: Path) -> None:
    command = tmp_path / "bin/nht-reconstruct"
    command.parent.mkdir()
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)
    request = _request(tmp_path)

    absolute_request = ReconstructionCommandRequest(
        scene_id=request.scene_id,
        input_video=request.input_video,
        workspace=request.workspace,
        pipeline_config=request.pipeline_config,
        executable=command.resolve(),
    )

    assert absolute_request.argv()[0] == str(command.resolve())


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("schema: [\n", "valid YAML"),
        ("- nht_pipeline_config_v1\n", "string-keyed mapping"),
        ("schema: legacy_nht_config\n", "schema"),
        (
            f"schema: {NHT_PIPELINE_CONFIG_SCHEMA}\nprivate_runtime: true\n",
            "Unknown NHT pipeline config key",
        ),
    ],
)
def test_pipeline_config_rejects_invalid_public_envelopes(
    tmp_path: Path,
    contents: str,
    error: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        NHTPipelineConfig.load(path.resolve())


def test_request_revalidates_pipeline_config_before_use(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.pipeline_config.path.write_text("schema: invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        ReconstructionCommandRequest(
            scene_id=request.scene_id,
            input_video=request.input_video,
            workspace=request.workspace,
            pipeline_config=request.pipeline_config,
        )


def test_runner_rejects_private_subprocess_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported private key"):
        nht_subprocess.run_nht_reconstruction(
            _request(tmp_path),
            environment={"PYTHONPATH": "/provider/private/modules"},
        )


def test_runner_rejects_command_success_without_public_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    validation_called = False

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        request.workspace.mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def unexpected_validation(_path: Path) -> StandardSceneExport:
        nonlocal validation_called
        validation_called = True
        raise AssertionError("scene validation must follow the public run manifest gate")

    monkeypatch.setattr(nht_subprocess.subprocess, "run", fake_run)
    monkeypatch.setattr(
        nht_subprocess,
        "validate_standard_scene_export",
        unexpected_validation,
    )

    with pytest.raises(FileNotFoundError, match="fixed run.json"):
        nht_subprocess.run_nht_reconstruction(request)

    assert validation_called is False


def test_runner_uses_shell_free_subprocess_and_validates_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    training_runtime = _training_runtime(tmp_path)
    recorded: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["argv"] = argv
        config_path = Path(argv[argv.index("--config") + 1])
        recorded["effective_config"] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        recorded.update(kwargs)
        request.workspace.mkdir(parents=True)
        request.run_manifest_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    identity = tuple(float(value) for value in np.eye(4).ravel())
    scene = StandardSceneExport(
        scene_id="B00",
        export_root=request.scene_path.parent,
        scene_path=request.scene_path,
        cameras=(),
        points_scene=np.empty((0, 6), dtype=np.float32),
        scene_from_sfm=identity,
        sfm_from_scene=identity,
        checkpoint_path=request.scene_path.parent / "model/ckpts/model.pt",
        runtime_config_path=request.scene_path.parent / "model/runtime-config.json",
    )
    monkeypatch.setattr(nht_subprocess.subprocess, "run", fake_run)
    monkeypatch.setattr(
        nht_subprocess, "validate_standard_scene_export", lambda _path: scene
    )

    result = nht_subprocess.run_nht_reconstruction(
        request,
        training_runtime=training_runtime,
    )

    assert result is scene
    argv = recorded["argv"]
    assert isinstance(argv, list)
    assert argv[:-1] == list(request.argv())[:-1]
    assert argv[-1] != str(request.pipeline_config.path)
    effective_config = recorded["effective_config"]
    assert isinstance(effective_config, dict)
    assert effective_config["nht_training"] == {
        "python": str(training_runtime.python),
        "trainer": str(training_runtime.trainer),
    }
    assert request.pipeline_config.path.read_text(encoding="utf-8") == (
        f"schema: {NHT_PIPELINE_CONFIG_SCHEMA}\n"
    )
    assert recorded["shell"] is False
    assert recorded["check"] is True
