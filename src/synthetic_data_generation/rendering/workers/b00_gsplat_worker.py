"""Standalone B00 checkpoint renderer; imports no tennis-lab or provider app code."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gsplat.rendering import rasterization

REQUEST_SCHEMA = "b00_gsplat_static_render_request_v1"
RESPONSE_SCHEMA = "b00_gsplat_static_render_response_v1"
BACKEND_ID = "b00-gsplat-file-subprocess"
DEPTH_CONVENTION = "opencv_camera_z_scene_units"


def main() -> None:
    """Render one request supplied as two exact positional paths."""
    if len(sys.argv) != 3:
        raise SystemExit("Usage: b00_gsplat_worker.py REQUEST_JSON RESPONSE_JSON")
    request_path = Path(sys.argv[1]).resolve()
    response_path = Path(sys.argv[2]).resolve()
    if response_path.exists():
        raise FileExistsError(f"Refusing to overwrite response: {response_path}")
    request = _load_request(request_path)
    checkpoint = request["checkpoint"]
    camera = request["camera"]
    render = request["render"]
    checkpoint_path = Path(checkpoint["path"]).resolve()
    actual_checkpoint_sha = _sha256_file(checkpoint_path)
    if actual_checkpoint_sha != checkpoint["sha256"]:
        raise ValueError("B00 checkpoint SHA-256 mismatch.")

    started = time.perf_counter()
    device = torch.device(render["device"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(state) != {"step", "scene_id", "splats"}:
        raise ValueError("B00 checkpoint top-level keys are unsupported.")
    if int(state["step"]) != 29999:
        raise ValueError(f"Expected final B00 step 29999, got {state['step']}.")
    splats = state["splats"]
    expected_splats = {"means", "opacities", "quats", "scales", "sh0", "shN"}
    if set(splats) != expected_splats:
        raise ValueError("B00 splat tensor keys are unsupported.")

    means = splats["means"].to(device, non_blocking=False)
    quats = splats["quats"].to(device, non_blocking=False)
    scales = torch.exp(splats["scales"].to(device, non_blocking=False))
    opacities = torch.sigmoid(splats["opacities"].to(device, non_blocking=False))
    colors = torch.cat((splats["sh0"], splats["shN"]), dim=1).to(
        device,
        non_blocking=False,
    )
    camera_to_scene = torch.tensor(
        camera["camera_to_scene"],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 4, 4)
    viewmat = torch.linalg.inv(camera_to_scene)
    intrinsics = torch.tensor(
        camera["intrinsics"],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 3)
    background = torch.tensor(
        render["background_rgb"],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3)
    with torch.inference_mode():
        rendered, rendered_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=int(camera["width"]),
            height=int(camera["height"]),
            packed=bool(render["packed"]),
            near_plane=float(render["near_plane"]),
            far_plane=float(render["far_plane"]),
            render_mode=str(render["render_mode"]),
            sh_degree=int(render["sh_degree"]),
            rasterize_mode=str(render["rasterize_mode"]),
            camera_model=str(render["camera_model"]),
            backgrounds=background,
        )
        torch.cuda.synchronize(device)
    rgb_float = rendered[0, ..., :3].clamp(0.0, 1.0)
    depth_tensor = rendered[0, ..., 3]
    alpha_tensor = rendered_alpha[0, ..., 0].clamp(0.0, 1.0)
    empty = alpha_tensor <= float(render["empty_alpha_threshold"])
    depth_tensor = torch.where(
        empty,
        torch.full_like(depth_tensor, torch.inf),
        depth_tensor,
    )
    if bool(
        torch.any((~empty) & ((~torch.isfinite(depth_tensor)) | (depth_tensor <= 0)))
    ):
        raise ValueError("B00 renderer returned invalid non-empty camera-Z depth.")
    rgb = torch.round(rgb_float * 255.0).to(torch.uint8).cpu().numpy()
    depth = depth_tensor.to(torch.float32).cpu().numpy()
    alpha = alpha_tensor.to(torch.float32).cpu().numpy()

    output_path = response_path.parent / "scene_frame.npz"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite payload: {output_path}")
    np.savez_compressed(output_path, rgb=rgb, depth=depth, alpha=alpha)
    response = {
        "schema": RESPONSE_SCHEMA,
        "request_fingerprint": request["request_fingerprint"],
        "scene_fingerprint": request["scene_fingerprint"],
        "camera_id": camera["camera_id"],
        "backend_id": BACKEND_ID,
        "backend_version": request["backend"]["source_revision"],
        "depth_convention": DEPTH_CONVENTION,
        "output": {
            "filename": output_path.name,
            "sha256": _sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "gaussian_count": int(means.shape[0]),
            "checkpoint_step": int(state["step"]),
        },
    }
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("B00 request must be a JSON object.")
    expected_keys = {
        "schema",
        "request_fingerprint",
        "scene_fingerprint",
        "camera",
        "checkpoint",
        "backend",
        "render",
    }
    if set(request) != expected_keys or request["schema"] != REQUEST_SCHEMA:
        raise ValueError("B00 request schema or keys are unsupported.")
    declared = request["request_fingerprint"]
    unsigned = dict(request)
    del unsigned["request_fingerprint"]
    if _fingerprint_json(unsigned) != declared:
        raise ValueError("B00 request fingerprint mismatch.")
    if set(request["camera"]) != {
        "camera_id",
        "width",
        "height",
        "intrinsics",
        "camera_to_scene",
    }:
        raise ValueError("B00 camera record keys are unsupported.")
    if set(request["checkpoint"]) != {"path", "sha256"}:
        raise ValueError("B00 checkpoint record keys are unsupported.")
    if set(request["backend"]) != {
        "source_revision",
        "prebuilt_module_sha256",
        "worker_sha256",
        "wrapper_sha256",
    }:
        raise ValueError("B00 backend record keys are unsupported.")
    if set(request["render"]) != {
        "device",
        "render_mode",
        "camera_model",
        "rasterize_mode",
        "sh_degree",
        "near_plane",
        "far_plane",
        "packed",
        "background_rgb",
        "empty_alpha_threshold",
    }:
        raise ValueError("B00 render record keys are unsupported.")
    if request["render"] != {
        "device": request["render"]["device"],
        "render_mode": "RGB+ED",
        "camera_model": "pinhole",
        "rasterize_mode": "classic",
        "sh_degree": 3,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "packed": False,
        "background_rgb": [0.0, 0.0, 0.0],
        "empty_alpha_threshold": 1.0e-6,
    }:
        raise ValueError("B00 render settings are unsupported.")
    return request


def _fingerprint_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
