"""Unit tests for the strict B00 file/subprocess contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.rendering.gsplat_subprocess_adapter as adapter_module
from src.synthetic_data_generation.rendering.gsplat_subprocess_adapter import (
    BACKEND_ID,
    DEPTH_CONVENTION,
    RESPONSE_SCHEMA,
    GsplatSubprocessConfig,
    build_request,
    load_verified_response,
    sha256_file,
)


def test_config_rejects_tampered_worker(tmp_path: Path) -> None:
    paths = _runtime_files(tmp_path)

    with pytest.raises(ValueError, match="worker SHA-256 mismatch"):
        _config(paths, worker_sha256="0" * 64)


def test_adapter_has_no_external_application_import_or_sys_path_injection() -> None:
    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(name == "gsplat" or name.startswith("gsplat.") for name in imports)
    assert "sys" not in imports
    assert "gaussian_splating" not in source


def test_response_payload_is_bound_to_request(
    tmp_path: Path,
    scene_camera,
) -> None:
    paths = _runtime_files(tmp_path)
    config = _config(paths)
    request = build_request(
        scene_fingerprint="a" * 64,
        camera=scene_camera,
        config=config,
    )
    payload_path = tmp_path / "scene_frame.npz"
    np.savez_compressed(
        payload_path,
        rgb=np.zeros(
            (scene_camera.height, scene_camera.width, 3),
            dtype=np.uint8,
        ),
        depth=np.full(
            (scene_camera.height, scene_camera.width),
            np.inf,
            dtype=np.float32,
        ),
        alpha=np.zeros(
            (scene_camera.height, scene_camera.width),
            dtype=np.float32,
        ),
    )
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "schema": RESPONSE_SCHEMA,
                "request_fingerprint": request["request_fingerprint"],
                "scene_fingerprint": request["scene_fingerprint"],
                "camera_id": scene_camera.camera_id,
                "backend_id": BACKEND_ID,
                "backend_version": config.backend_source_revision,
                "depth_convention": DEPTH_CONVENTION,
                "output": {
                    "filename": payload_path.name,
                    "sha256": sha256_file(payload_path),
                    "size_bytes": payload_path.stat().st_size,
                },
                "runtime": {
                    "elapsed_seconds": 1.0,
                    "python": "3.12.3",
                    "torch": "2.10.0+cu130",
                    "cuda": "13.0",
                    "device_name": "test",
                    "gaussian_count": 1_286_654,
                    "checkpoint_step": 29999,
                },
            }
        ),
        encoding="utf-8",
    )

    frame = load_verified_response(
        response_path,
        request=request,
        expected_camera=scene_camera,
        backend_version=config.backend_source_revision,
    )

    assert frame.rgb.shape == (
        scene_camera.height,
        scene_camera.width,
        3,
    )
    assert frame.backend_id == BACKEND_ID

    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        load_verified_response(
            response_path,
            request=request,
            expected_camera=scene_camera,
            backend_version=config.backend_source_revision,
        )


def _runtime_files(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name in ("python", "wrapper", "module", "worker", "checkpoint"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        result[name] = path
    result["working_directory"] = tmp_path
    return result


def _config(
    paths: dict[str, Path],
    *,
    worker_sha256: str | None = None,
) -> GsplatSubprocessConfig:
    return GsplatSubprocessConfig(
        python_executable=paths["python"],
        wrapper_script=paths["wrapper"],
        wrapper_sha256=sha256_file(paths["wrapper"]),
        prebuilt_module=paths["module"],
        prebuilt_module_sha256=sha256_file(paths["module"]),
        worker_script=paths["worker"],
        worker_sha256=worker_sha256 or sha256_file(paths["worker"]),
        checkpoint=paths["checkpoint"],
        checkpoint_sha256=sha256_file(paths["checkpoint"]),
        backend_source_revision="b" * 40,
        working_directory=paths["working_directory"],
    )
