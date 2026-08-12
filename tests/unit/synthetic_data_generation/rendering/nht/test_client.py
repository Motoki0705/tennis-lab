from __future__ import annotations

import gc
import json
import subprocess
import weakref
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


def _camera(camera_id: str = "novel") -> NHTRenderCamera:
    return NHTRenderCamera(
        camera_id=camera_id,
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
    camera_ids: tuple[str, ...] = ("novel",),
) -> None:
    renders: list[dict[str, object]] = []
    for camera_id in camera_ids:
        frame = output / camera_id
        frame.mkdir(parents=True)
        np.save(frame / "rgb.npy", np.full((6, 8, 3), 0.5, dtype=dtype))
        np.save(frame / "alpha.npy", np.ones((6, 8, 1), dtype=dtype))
        np.save(frame / "depth.npy", np.ones((6, 8, 1), dtype=dtype))
        Image.new("RGB", (8, 6)).save(frame / "rgb.png")
        Image.new("L", (8, 6)).save(frame / "alpha.png")
        renders.append(
            {
                "camera_id": camera_id,
                "request_source": "arbitrary",
                "width": 8,
                "height": 6,
                "rgb": f"{camera_id}/rgb.npy",
                "rgb_preview": f"{camera_id}/rgb.png",
                "alpha": f"{camera_id}/alpha.npy",
                "alpha_preview": f"{camera_id}/alpha.png",
                "depth": f"{camera_id}/depth.npy",
            }
        )
    (output / "render.json").write_text(
        json.dumps(
            {
                "schema": "nht_render_result_v1",
                "scene_schema": "nht_standard_scene_v1",
                "scene_id": "B00",
                "coordinate_space": "canonical NHT scene space",
                "export_validation": {},
                "renders": renders,
            }
        ),
        encoding="utf-8",
    )


def _command(
    tmp_path: Path,
    *,
    output_name: str = "render",
    camera_ids: tuple[str, ...] = ("novel",),
) -> NHTRenderCommandRequest:
    scene_path = tmp_path / "B00/reconstruction/export/scene.json"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    if not scene_path.exists():
        scene_path.write_text("{}", encoding="utf-8")
    return NHTRenderCommandRequest(
        scene_path=scene_path,
        output_directory=tmp_path / f"datasets/court/staging/{output_name}",
        arbitrary_cameras=NHTRenderRequest(
            tuple(_camera(camera_id) for camera_id in camera_ids)
        ),
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
    assert tuple(item.path.name for item in record.array_metadata) == (
        "rgb.npy",
        "alpha.npy",
        "depth.npy",
    )
    assert tuple(item.shape for item in record.array_metadata) == (
        (6, 8, 3),
        (6, 8, 1),
        (6, 8, 1),
    )
    assert all(item.dtype == "float32" for item in record.array_metadata)
    arrays = record.load_arrays()
    assert arrays.rgb.shape == (6, 8, 3)
    assert not arrays.rgb.flags.writeable
    assert np.array_equal(
        arrays.metric_depth(nht_scene_units_per_metre=2.0),
        np.full((6, 8, 1), 0.5, dtype=np.float32),
    )
    assert result.evidence.invocation_index == 1
    assert result.evidence.scene_validation_count == 1
    assert result.evidence.scene_cache_hit is False
    assert result.evidence.complete_payload_scan_count == 1
    assert result.evidence.array_file_load_count == 3
    assert result.evidence.preview_validation_count == 2
    assert result.evidence.loaded_array_bytes == 6 * 8 * 5 * 4
    assert result.evidence.maximum_live_array_bytes == 6 * 8 * 5 * 4
    assert result.evidence.retained_array_bytes == 0
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
    assert first_result.record("novel").retained_array_byte_count == 0


def test_client_releases_dense_payloads_between_render_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _command(tmp_path, output_name="shard-0")
    second = _command(tmp_path, output_name="shard-1")
    references: list[weakref.ReferenceType[NDArray[np.generic]]] = []
    original_load = np.load

    def tracked_load(
        file: Path,
        *,
        allow_pickle: bool = False,
    ) -> NDArray[np.generic]:
        loaded = original_load(file, allow_pickle=allow_pickle)
        if not isinstance(loaded, np.ndarray):
            raise TypeError("NHT render arrays must be stored in .npy files.")
        references.append(weakref.ref(loaded))
        return loaded

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = Path(argv[argv.index("--output") + 1])
        _write_result(output)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        client, "validate_standard_scene_export", lambda path: _scene(path)
    )
    monkeypatch.setattr(client.np, "load", tracked_load)
    monkeypatch.setattr(client.subprocess, "run", fake_run)
    render_client = client.NHTRenderClient()

    first_result = render_client.render(first)
    gc.collect()
    assert len(references) == 3
    assert all(reference() is None for reference in references)

    second_result = render_client.render(second)
    gc.collect()
    assert len(references) == 6
    assert all(reference() is None for reference in references)
    assert first_result.evidence.retained_array_bytes == 0
    assert second_result.evidence.retained_array_bytes == 0


def test_client_bounds_live_payload_to_one_record_and_retains_exact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(5))
    command = _command(tmp_path, camera_ids=camera_ids)
    references: list[weakref.ReferenceType[NDArray[np.generic]]] = []
    maximum_live_loads = 0
    original_load = np.load

    def tracked_load(
        file: Path,
        *,
        allow_pickle: bool = False,
    ) -> NDArray[np.generic]:
        nonlocal maximum_live_loads
        loaded = original_load(file, allow_pickle=allow_pickle)
        if not isinstance(loaded, np.ndarray):
            raise TypeError("NHT render arrays must be stored in .npy files.")
        references.append(weakref.ref(loaded))
        maximum_live_loads = max(
            maximum_live_loads,
            sum(reference() is not None for reference in references),
        )
        return loaded

    monkeypatch.setattr(
        client, "validate_standard_scene_export", lambda path: _scene(path)
    )
    monkeypatch.setattr(client.np, "load", tracked_load)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        _write_result(command.output_directory, camera_ids=camera_ids)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(client.subprocess, "run", fake_run)

    result = client.NHTRenderClient().render(command)

    bytes_per_record = 6 * 8 * 5 * np.dtype(np.float32).itemsize
    assert tuple(record.camera_id for record in result.records) == camera_ids
    assert result.evidence.camera_count == len(camera_ids)
    assert result.evidence.loaded_array_bytes == len(camera_ids) * bytes_per_record
    assert result.evidence.maximum_live_array_bytes == bytes_per_record
    assert result.evidence.retained_array_bytes == 0
    assert maximum_live_loads == 3
    assert sum(record.retained_array_byte_count for record in result.records) == 0
    assert all(
        record.validated_array_byte_count == bytes_per_record
        for record in result.records
    )


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
