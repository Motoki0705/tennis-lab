"""Strict file/subprocess adapter for the external B00 gsplat scene provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.rendering.scene_frame_port import SceneFrame
from src.synthetic_data_generation.scene_contract import SceneCamera

REQUEST_SCHEMA = "b00_gsplat_static_render_request_v1"
RESPONSE_SCHEMA = "b00_gsplat_static_render_response_v1"
BACKEND_ID = "b00-gsplat-file-subprocess"
DEPTH_CONVENTION = "opencv_camera_z_scene_units"
_REQUEST_KEYS = {
    "schema",
    "request_fingerprint",
    "scene_fingerprint",
    "camera",
    "checkpoint",
    "backend",
    "render",
}
_RESPONSE_KEYS = {
    "schema",
    "request_fingerprint",
    "scene_fingerprint",
    "camera_id",
    "backend_id",
    "backend_version",
    "depth_convention",
    "output",
    "runtime",
}
_RUNTIME_KEYS = {
    "elapsed_seconds",
    "python",
    "torch",
    "cuda",
    "device_name",
    "gaussian_count",
    "checkpoint_step",
}
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class GsplatSubprocessConfig:
    """All provider/runtime paths and identities required by the adapter."""

    python_executable: Path
    wrapper_script: Path
    wrapper_sha256: str
    prebuilt_module: Path
    prebuilt_module_sha256: str
    worker_script: Path
    worker_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    backend_source_revision: str
    working_directory: Path
    timeout_seconds: int = 300
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        for name, path in (
            ("python_executable", self.python_executable),
            ("wrapper_script", self.wrapper_script),
            ("prebuilt_module", self.prebuilt_module),
            ("worker_script", self.worker_script),
            ("checkpoint", self.checkpoint),
        ):
            if not path.is_file():
                raise ValueError(f"{name} is not a file: {path}")
        if not self.working_directory.is_dir():
            raise ValueError(
                f"working_directory is not a directory: {self.working_directory}"
            )
        for name, path, expected in (
            ("wrapper", self.wrapper_script, self.wrapper_sha256),
            ("prebuilt module", self.prebuilt_module, self.prebuilt_module_sha256),
            ("worker", self.worker_script, self.worker_sha256),
            ("checkpoint", self.checkpoint, self.checkpoint_sha256),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"{name} SHA-256 mismatch: expected {expected}, got {actual}."
                )
        if _GIT_SHA_PATTERN.fullmatch(self.backend_source_revision) is None:
            raise ValueError("backend_source_revision must be a 40-character git hash.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not self.device.startswith("cuda:"):
            raise ValueError("B00 gsplat rendering requires an explicit CUDA device.")


class B00GsplatSubprocessAdapter:
    """Render B00 through verified files without importing external application code."""

    def __init__(self, config: GsplatSubprocessConfig) -> None:
        self._config = config

    def render_scene_frame(
        self,
        *,
        scene_fingerprint: str,
        camera: SceneCamera,
    ) -> SceneFrame:
        """Render into a temporary exchange directory and strictly verify output."""
        with tempfile.TemporaryDirectory(prefix="b00-gsplat-render-") as temp_dir:
            exchange_dir = Path(temp_dir)
            request_path = exchange_dir / "request.json"
            response_path = exchange_dir / "response.json"
            request = build_request(
                scene_fingerprint=scene_fingerprint,
                camera=camera,
                config=self._config,
            )
            _write_json(request_path, request)
            command = [
                str(self._config.python_executable),
                str(self._config.wrapper_script),
                "--module",
                str(self._config.prebuilt_module),
                "--sha256",
                self._config.prebuilt_module_sha256,
                str(self._config.worker_script),
                str(request_path),
                str(response_path),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = self._config.device.removeprefix(
                "cuda:"
            )
            completed = subprocess.run(
                command,
                cwd=self._config.working_directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "B00 gsplat worker failed "
                    f"(exit {completed.returncode}).\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            if not response_path.is_file():
                raise RuntimeError("B00 gsplat worker did not publish response.json.")
            return load_verified_response(
                response_path,
                request=request,
                expected_camera=camera,
                backend_version=self._config.backend_source_revision,
            )

    def render_to_directory(
        self,
        *,
        scene_fingerprint: str,
        camera: SceneCamera,
        output_dir: Path,
    ) -> SceneFrame:
        """Atomically publish a verified immutable exchange artifact."""
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite render artifact: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
        )
        try:
            request = build_request(
                scene_fingerprint=scene_fingerprint,
                camera=camera,
                config=self._config,
            )
            request_path = staging / "request.json"
            response_path = staging / "response.json"
            _write_json(request_path, request)
            command = [
                str(self._config.python_executable),
                str(self._config.wrapper_script),
                "--module",
                str(self._config.prebuilt_module),
                "--sha256",
                self._config.prebuilt_module_sha256,
                str(self._config.worker_script),
                str(request_path),
                str(response_path),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = self._config.device.removeprefix(
                "cuda:"
            )
            completed = subprocess.run(
                command,
                cwd=self._config.working_directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )
            (staging / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (staging / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    "B00 gsplat worker failed "
                    f"(exit {completed.returncode}).\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            frame = load_verified_response(
                response_path,
                request=request,
                expected_camera=camera,
                backend_version=self._config.backend_source_revision,
            )
            staging.rename(output_dir)
            return frame
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def build_request(
    *,
    scene_fingerprint: str,
    camera: SceneCamera,
    config: GsplatSubprocessConfig,
) -> dict[str, object]:
    """Build a canonical worker request bound to exact artifacts and paths."""
    body: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "scene_fingerprint": scene_fingerprint,
        "camera": {
            "camera_id": camera.camera_id,
            "width": camera.width,
            "height": camera.height,
            "intrinsics": list(camera.intrinsics),
            "camera_to_scene": list(camera.camera_to_scene),
        },
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "sha256": config.checkpoint_sha256,
        },
        "backend": {
            "source_revision": config.backend_source_revision,
            "prebuilt_module_sha256": config.prebuilt_module_sha256,
            "worker_sha256": config.worker_sha256,
            "wrapper_sha256": config.wrapper_sha256,
        },
        "render": {
            "device": config.device,
            "render_mode": "RGB+ED",
            "camera_model": "pinhole",
            "rasterize_mode": "classic",
            "sh_degree": 3,
            "near_plane": 0.01,
            "far_plane": 1.0e10,
            "packed": False,
            "background_rgb": [0.0, 0.0, 0.0],
            "empty_alpha_threshold": 1.0e-6,
        },
    }
    body["request_fingerprint"] = fingerprint_json(body)
    if set(body) != _REQUEST_KEYS:
        raise AssertionError("Internal B00 request schema drift.")
    return body


def load_verified_response(
    response_path: Path,
    *,
    request: dict[str, object],
    expected_camera: SceneCamera,
    backend_version: str,
) -> SceneFrame:
    """Load a response, verify its binding and payload, and return typed arrays."""
    response = _load_mapping(response_path)
    if set(response) != _RESPONSE_KEYS:
        raise ValueError("B00 response keys do not match the v1 schema.")
    if response["schema"] != RESPONSE_SCHEMA:
        raise ValueError(f"Unsupported B00 response schema: {response['schema']!r}.")
    for key in ("request_fingerprint", "scene_fingerprint"):
        if response[key] != request[key]:
            raise ValueError(f"B00 response {key} does not match request.")
    if response["camera_id"] != expected_camera.camera_id:
        raise ValueError("B00 response camera_id does not match request.")
    if response["backend_id"] != BACKEND_ID:
        raise ValueError("B00 response backend_id is unsupported.")
    if response["backend_version"] != backend_version:
        raise ValueError("B00 response backend version mismatch.")
    if response["depth_convention"] != DEPTH_CONVENTION:
        raise ValueError("B00 response depth convention mismatch.")
    runtime = response["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS:
        raise ValueError("B00 response runtime record is invalid.")
    if runtime["gaussian_count"] != 1_286_654:
        raise ValueError("B00 response Gaussian count mismatch.")
    if runtime["checkpoint_step"] != 29999:
        raise ValueError("B00 response checkpoint step mismatch.")
    output = response["output"]
    if not isinstance(output, dict) or set(output) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("B00 response output record is invalid.")
    output_path = response_path.parent / str(output["filename"])
    if not output_path.is_file():
        raise ValueError("B00 response payload is missing.")
    if output_path.stat().st_size != output["size_bytes"]:
        raise ValueError("B00 response payload size mismatch.")
    if sha256_file(output_path) != output["sha256"]:
        raise ValueError("B00 response payload SHA-256 mismatch.")
    with np.load(output_path, allow_pickle=False) as payload:
        if set(payload.files) != {"rgb", "depth", "alpha"}:
            raise ValueError("B00 response arrays do not match the v1 schema.")
        rgb = np.asarray(payload["rgb"])
        depth = np.asarray(payload["depth"])
        alpha = np.asarray(payload["alpha"])
    expected_shape = (expected_camera.height, expected_camera.width)
    if rgb.shape != (*expected_shape, 3) or depth.shape != expected_shape:
        raise ValueError("B00 response array dimensions differ from SceneCamera.")
    if alpha.shape != expected_shape:
        raise ValueError("B00 response alpha dimensions differ from SceneCamera.")
    return SceneFrame(
        rgb=rgb,
        depth=depth,
        alpha=alpha,
        scene_fingerprint=str(response["scene_fingerprint"]),
        camera_id=str(response["camera_id"]),
        backend_id=str(response["backend_id"]),
        backend_version=str(response["backend_version"]),
    )


def fingerprint_json(value: object) -> str:
    """Return SHA-256 of a canonical compact JSON value."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return raw


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
