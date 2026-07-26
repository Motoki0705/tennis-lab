"""
Render one accepted B00 captured camera through the file/subprocess adapter.

Usage:
    python -m src.synthetic_data_generation.scripts.render_b00_static_smoke

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/render_b00_static_smoke.yaml`.
    - The external provider repository is only read through verified artifacts.
    - Publication refuses overwrite and uses an atomic directory rename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.synthetic_data_generation.rendering.gsplat_subprocess_adapter import (
    B00GsplatSubprocessAdapter,
    GsplatSubprocessConfig,
    build_request,
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="render_b00_static_smoke",
)
def main(cfg: DictConfig) -> int:
    """Verify identities and publish one real B00 scene-frame render."""
    scene_contract_path = _path(cfg.scene_contract)
    expected_contract_sha = str(cfg.scene_contract_sha256)
    actual_contract_sha = sha256_file(scene_contract_path)
    if actual_contract_sha != expected_contract_sha:
        raise ValueError(
            "Scene contract SHA-256 mismatch: "
            f"expected {expected_contract_sha}, got {actual_contract_sha}."
        )
    contract = load_scene_contract(scene_contract_path)
    if contract.scene_fingerprint != str(cfg.scene_fingerprint):
        raise ValueError("Scene fingerprint differs from the frozen smoke config.")
    camera_id = str(cfg.camera_id)
    matches = [camera for camera in contract.cameras if camera.camera_id == camera_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one camera {camera_id!r}.")
    camera = matches[0]

    adapter_config = GsplatSubprocessConfig(
        python_executable=_executable_path(cfg.provider.python_executable),
        wrapper_script=_path(cfg.provider.wrapper_script),
        wrapper_sha256=str(cfg.provider.wrapper_sha256),
        prebuilt_module=_path(cfg.provider.prebuilt_module),
        prebuilt_module_sha256=str(cfg.provider.prebuilt_module_sha256),
        worker_script=_path(cfg.provider.worker_script),
        worker_sha256=str(cfg.provider.worker_sha256),
        checkpoint=_path(cfg.provider.checkpoint),
        checkpoint_sha256=str(cfg.provider.checkpoint_sha256),
        backend_source_revision=str(cfg.provider.backend_source_revision),
        working_directory=_path(cfg.provider.working_directory),
        timeout_seconds=int(cfg.provider.timeout_seconds),
        device=str(cfg.provider.device),
    )
    request = build_request(
        scene_fingerprint=contract.scene_fingerprint,
        camera=camera,
        config=adapter_config,
    )
    output_dir = _path(cfg.output_root) / (
        f"{camera.camera_id}-{request['request_fingerprint']}"
    )
    frame = B00GsplatSubprocessAdapter(adapter_config).render_to_directory(
        scene_fingerprint=contract.scene_fingerprint,
        camera=camera,
        output_dir=output_dir,
    )
    response = json.loads((output_dir / "response.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "request_fingerprint": request["request_fingerprint"],
        "camera_id": frame.camera_id,
        "shape": list(frame.rgb.shape),
        "finite_depth_pixels": int((frame.depth < float("inf")).sum()),
        "nonzero_alpha_pixels": int((frame.alpha > 0.0).sum()),
        "alpha_mean": float(frame.alpha.mean()),
        "runtime": response["runtime"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _executable_path(value: Any) -> Path:
    """Return an absolute executable path without resolving a venv symlink."""
    return Path(to_absolute_path(str(value))).absolute()


if __name__ == "__main__":
    raise SystemExit(main())
