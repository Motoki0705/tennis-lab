from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import client
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _camera() -> NHTRenderCamera:
    return NHTRenderCamera(
        camera_id="novel",
        width=8,
        height=6,
        intrinsics=(5.0, 0.0, 4.0, 0.0, 5.0, 3.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(np.eye(4)),
    )


def _scene(scene_path: Path) -> StandardSceneExport:
    (scene_path.parent / "cameras.json").write_text("{}", encoding="utf-8")
    np.save(
        scene_path.parent / "points_scene.npy",
        np.empty((0, 6), dtype=np.float32),
    )
    model_root = scene_path.parent / "model"
    model_root.mkdir(exist_ok=True)
    checkpoint_path = model_root / "model.pt"
    runtime_config_path = model_root / "runtime.json"
    if not checkpoint_path.exists():
        checkpoint_path.write_bytes(b"model")
    if not runtime_config_path.exists():
        runtime_config_path.write_text("{}", encoding="utf-8")
    points: NDArray[np.float32] = np.empty((0, 6), dtype=np.float32)
    points.setflags(write=False)
    return StandardSceneExport(
        scene_id="B00",
        export_root=scene_path.parent,
        scene_path=scene_path,
        cameras=(),
        points_scene=points,
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=checkpoint_path,
        runtime_config_path=runtime_config_path,
    )


def _write_result(
    output: Path,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> None:
    frame = output / "novel"
    frame.mkdir(parents=True)
    np.save(frame / "rgb.npy", np.full((6, 8, 3), 0.5, dtype=dtype))
    np.save(frame / "alpha.npy", np.ones((6, 8, 1), dtype=dtype))
    np.save(frame / "depth.npy", np.ones((6, 8, 1), dtype=dtype))
    Image.new("RGB", (8, 6)).save(frame / "rgb.png")
    Image.new("L", (8, 6)).save(frame / "alpha.png")
    (output / "render.json").write_text(
        json.dumps(
            {
                "schema": "nht_render_result_v1",
                "scene_schema": "nht_standard_scene_v1",
                "scene_id": "B00",
                "coordinate_space": "canonical NHT scene space",
                "export_validation": {},
                "renders": [
                    {
                        "camera_id": "novel",
                        "request_source": "arbitrary",
                        "width": 8,
                        "height": 6,
                        "rgb": "novel/rgb.npy",
                        "rgb_preview": "novel/rgb.png",
                        "alpha": "novel/alpha.npy",
                        "alpha_preview": "novel/alpha.png",
                        "depth": "novel/depth.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _command(
    tmp_path: Path,
    *,
    output_name: str = "render",
) -> NHTRenderCommandRequest:
    scene_path = tmp_path / "B00/reconstruction/export/scene.json"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    if not scene_path.exists():
        scene_path.write_text("{}", encoding="utf-8")
    return NHTRenderCommandRequest(
        scene_path=scene_path,
        output_directory=tmp_path / f"datasets/court/staging/{output_name}",
        arbitrary_cameras=NHTRenderRequest((_camera(),)),
        arbitrary_request_path=tmp_path
        / f"datasets/court/staging/{output_name}-request.json",
    )


def test_client_uses_file_boundary_and_validates_all_result_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        client, "validate_standard_scene_export", lambda path: _scene(path)
    )

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["argv"] = argv
        recorded.update(kwargs)
        assert command.arbitrary_request_path is not None
        assert json.loads(command.arbitrary_request_path.read_text(encoding="utf-8"))[
            "schema"
        ] == ("nht_render_request_v1")
        _write_result(command.output_directory)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client.subprocess, "run", fake_run)

    result = client.NHTRenderClient().render(command)

    assert result.scene_id == "B00"
    record = result.record("novel")
    assert record.rgb_path.is_file()
    assert record.arrays.rgb.shape == (6, 8, 3)
    assert not record.arrays.rgb.flags.writeable
    assert np.array_equal(
        record.arrays.metric_depth(nht_scene_units_per_metre=2.0),
        np.full((6, 8, 1), 0.5, dtype=np.float32),
    )
    assert result.evidence.invocation_index == 1
    assert result.evidence.scene_validation_count == 1
    assert result.evidence.scene_cache_hit is False
    assert result.evidence.complete_payload_scan_count == 1
    assert result.evidence.array_file_load_count == 3
    assert result.evidence.preview_validation_count == 2
    assert result.evidence.loaded_array_bytes == 6 * 8 * 5 * 4
    assert recorded["argv"] == list(command.argv())
    assert recorded["shell"] is False


def test_client_reuses_unchanged_scene_validation_and_scans_each_result_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _command(tmp_path, output_name="render-0")
    second = _command(tmp_path, output_name="render-1")
    validation_count = 0
    array_load_count = 0
    original_load = np.load

    def validate(path: Path) -> StandardSceneExport:
        nonlocal validation_count
        validation_count += 1
        return _scene(path)

    def load(file: Path, *, allow_pickle: bool = False) -> NDArray[np.generic]:
        nonlocal array_load_count
        array_load_count += 1
        loaded = original_load(file, allow_pickle=allow_pickle)
        if not isinstance(loaded, np.ndarray):
            raise TypeError("NHT render arrays must be stored in .npy files.")
        return loaded

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = Path(argv[argv.index("--output") + 1])
        _write_result(output)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client, "validate_standard_scene_export", validate)
    monkeypatch.setattr(client.np, "load", load)
    monkeypatch.setattr(client.subprocess, "run", fake_run)
    render_client = client.NHTRenderClient()

    scene = render_client.validate_scene(first.scene_path)
    first_result = render_client.render(first)
    second_result = render_client.render(second)

    assert scene.scene_id == "B00"
    assert validation_count == 1
    assert array_load_count == 6
    assert first_result.evidence.scene_cache_hit is True
    assert first_result.evidence.scene_validation_count == 0
    assert second_result.evidence.scene_cache_hit is True
    assert second_result.evidence.scene_validation_count == 0
    assert second_result.evidence.invocation_index == 2
    assert first_result.record("novel").arrays is first_result.record("novel").arrays


def test_client_invalidates_cached_scene_when_a_validated_dependency_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _command(tmp_path, output_name="render-0")
    second = _command(tmp_path, output_name="render-1")
    validation_count = 0

    def validate(path: Path) -> StandardSceneExport:
        nonlocal validation_count
        validation_count += 1
        return _scene(path)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = Path(argv[argv.index("--output") + 1])
        _write_result(output)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client, "validate_standard_scene_export", validate)
    monkeypatch.setattr(client.subprocess, "run", fake_run)
    render_client = client.NHTRenderClient()
    render_client.render(first)
    runtime_config = first.scene_path.parent / "model/runtime.json"
    runtime_config.write_text('{"changed": true}', encoding="utf-8")

    result = render_client.render(second)

    assert validation_count == 2
    assert result.evidence.scene_cache_hit is False
    assert result.evidence.scene_validation_count == 1


def test_client_fails_closed_when_changed_scene_dependency_no_longer_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _command(tmp_path, output_name="render-0")
    second = _command(tmp_path, output_name="render-1")
    validation_count = 0
    invocation_count = 0

    def validate(path: Path) -> StandardSceneExport:
        nonlocal validation_count
        validation_count += 1
        if validation_count > 1:
            raise ValueError("changed scene dependency is invalid")
        return _scene(path)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal invocation_count
        invocation_count += 1
        output = Path(argv[argv.index("--output") + 1])
        _write_result(output)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client, "validate_standard_scene_export", validate)
    monkeypatch.setattr(client.subprocess, "run", fake_run)
    render_client = client.NHTRenderClient()
    render_client.render(first)
    (first.scene_path.parent / "model/model.pt").write_bytes(b"changed model")

    with pytest.raises(ValueError, match="changed scene dependency is invalid"):
        render_client.render(second)

    assert validation_count == 2
    assert invocation_count == 1
    assert not second.output_directory.exists()


def test_client_rejects_non_float32_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    monkeypatch.setattr(
        client, "validate_standard_scene_export", lambda path: _scene(path)
    )

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        _write_result(command.output_directory, dtype=np.float64)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client.subprocess, "run", fake_run)

    with pytest.raises(TypeError, match="dtype float32"):
        client.NHTRenderClient().render(command)
