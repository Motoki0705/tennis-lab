from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.reconstruction import nht_subprocess
from src.synthetic_data_generation.reconstruction.contracts import (
    ReconstructionCommandRequest,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)


def _request(tmp_path: Path) -> ReconstructionCommandRequest:
    scene_root = tmp_path / "scenes/B00"
    source = scene_root / "source/video.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    return ReconstructionCommandRequest(
        scene_id="B00",
        input_video=source,
        workspace=scene_root / "reconstruction",
    )


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
    )
    assert "--config" not in request.argv()
    assert request.scene_path == request.workspace / "export/scene.json"


def test_request_rejects_noncanonical_workspace(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="fixed <scene_id>/reconstruction"):
        ReconstructionCommandRequest(
            scene_id=request.scene_id,
            input_video=request.input_video,
            workspace=tmp_path / "runs/attempt-1",
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
        executable=command.resolve(),
    )

    assert absolute_request.argv()[0] == str(command.resolve())


def test_runner_rejects_private_subprocess_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported private key"):
        nht_subprocess.run_nht_reconstruction(
            _request(tmp_path),
            environment={"PYTHONPATH": "/provider/private/modules"},
        )


def test_runner_uses_shell_free_subprocess_and_validates_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    recorded: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["argv"] = argv
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

    result = nht_subprocess.run_nht_reconstruction(request)

    assert result is scene
    assert recorded["argv"] == list(request.argv())
    assert recorded["shell"] is False
    assert recorded["check"] is True
