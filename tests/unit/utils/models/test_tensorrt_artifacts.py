"""Fail-closed engine manifests without a TensorRT or CUDA installation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.utils.models.tensorrt_artifacts import (
    ENGINE_MANIFEST,
    EXPORT_MANIFEST,
    ONNX_FILE,
    PLAN_FILE,
    EngineBundle,
    EngineEnvironment,
)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    for name, body in {
        ONNX_FILE: b"exported FP16 graph",
        f"{ONNX_FILE}.data": b"external weights",
        PLAN_FILE: b"compiled model plan",
    }.items():
        (tmp_path / name).write_bytes(body)
    graph = {
        "version": 1,
        "model": "example",
        "precision": "fp16",
        "settings": {"variant": "v1"},
        "sources": {"checkpoint": "b" * 64},
        "onnx_files": {
            name: file_hash(tmp_path / name)
            for name in [ONNX_FILE, f"{ONNX_FILE}.data"]
        },
        "inputs": {"pixels": {"shape": [-1, 3, 32, 64], "dtype": "float16"}},
        "outputs": {"features": {"shape": [-1, 512], "dtype": "float16"}},
    }
    (tmp_path / EXPORT_MANIFEST).write_text(json.dumps(graph))
    engine = {
        "version": 1,
        "export_sha256": file_hash(tmp_path / EXPORT_MANIFEST),
        "plan_sha256": file_hash(tmp_path / PLAN_FILE),
        "environment": {
            "tensorrt_version": "10.14.1",
            "device_name": "GPU",
            "compute_capability": [12, 0],
        },
        "profiles": {
            "pixels": {
                "minimum": [1, 3, 32, 64],
                "optimum": [2, 3, 32, 64],
                "maximum": [4, 3, 32, 64],
            },
        },
    }
    (tmp_path / ENGINE_MANIFEST).write_text(json.dumps(engine))
    return tmp_path


@pytest.mark.parametrize(
    "name", [ONNX_FILE, f"{ONNX_FILE}.data", PLAN_FILE, EXPORT_MANIFEST]
)
def test_changed_artifact_cannot_open(bundle_path: Path, name: str) -> None:
    EngineBundle.read(bundle_path)
    path = bundle_path / name
    content = path.read_bytes()
    path.write_bytes(b"!" + content[1:])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EngineBundle.read(bundle_path)


def test_missing_plan_cannot_open(bundle_path: Path) -> None:
    (bundle_path / PLAN_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="Missing TensorRT artifact"):
        EngineBundle.read(bundle_path)


@pytest.mark.parametrize(
    "changed",
    [
        {"tensorrt_version": "10.15.0"},
        {"device_name": "different GPU"},
        {"compute_capability": (8, 9)},
    ],
)
def test_incompatible_environment_cannot_load(
    bundle_path: Path, changed: dict[str, object]
) -> None:
    engine = EngineBundle.read(bundle_path)
    running_on = EngineEnvironment.model_validate(
        {
            "tensorrt_version": "10.14.1",
            "device_name": "GPU",
            "compute_capability": (12, 0),
        }
        | changed
    )
    with pytest.raises(ValueError, match="environment mismatch"):
        engine.require_environment(running_on)


def test_profile_must_preserve_export_shape(bundle_path: Path) -> None:
    path = bundle_path / ENGINE_MANIFEST
    engine = json.loads(path.read_text())
    engine["profiles"]["pixels"]["maximum"] = [4, 3, 64, 64]
    path.write_text(json.dumps(engine))
    with pytest.raises(ValueError, match="differs from ONNX shape"):
        EngineBundle.read(bundle_path)


def test_profile_bounds_include_tail_and_reject_oversize(bundle_path: Path) -> None:
    profile = EngineBundle.read(bundle_path).engine.profiles["pixels"]
    profile.require((1, 3, 32, 64), tensor="pixels")
    profile.require((4, 3, 32, 64), tensor="pixels")
    with pytest.raises(ValueError, match="outside profile"):
        profile.require((5, 3, 32, 64), tensor="pixels")
