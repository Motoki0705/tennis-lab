"""
Publish a deterministic BLCS-ball positive/negative pilot over one B00 frame.

Usage:
    python -m src.synthetic_data_generation.scripts.publish_b00_ball_pilot

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/publish_b00_ball_pilot.yaml`.
    - BLCS generates court-metre physics; the accepted SceneContract applies Sim(3) once.
    - The output is TrackNet-compatible, fingerprinted, atomic, and training-only.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.synthetic_data_generation.dataset.single_frame_pilot import (
    PilotProvenance,
    SingleFramePilotConfig,
    load_and_validate_single_frame_pilot,
    publish_single_frame_pilot,
)
from src.synthetic_data_generation.rendering.cpu_fake_renderer import (
    CpuSceneFrame,
    DeterministicCpuSphereRenderer,
)
from src.synthetic_data_generation.rendering.gsplat_subprocess_adapter import (
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneGenerator
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="publish_b00_ball_pilot",
)
def main(cfg: DictConfig) -> int:
    """Generate the frozen BLCS rally and publish two supervised pilot frames."""
    repo_root = Path(to_absolute_path(".")).resolve()
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    contract_path = _verified_file(
        cfg.scene_contract.path,
        expected_sha256=str(cfg.scene_contract.sha256),
    )
    contract = load_scene_contract(contract_path)
    if contract.scene_fingerprint != str(cfg.scene_contract.scene_fingerprint):
        raise ValueError("Pilot SceneContract fingerprint mismatch.")

    scene = BLCSSceneGenerator(device="cpu").generate_scene(
        int(cfg.blcs.from_cell),
        str(cfg.blcs.side),
        str(cfg.blcs.scene_id),
    )
    if scene is None:
        raise RuntimeError("Frozen BLCS pilot generation returned no scene.")
    if int(scene.ball_pos_world.shape[0]) != int(cfg.blcs.expected_frame_count):
        raise ValueError("Frozen BLCS pilot trajectory frame count changed.")
    selected_index = int(cfg.pilot.trajectory_frame_index)
    actual_position = scene.ball_pos_world[selected_index].detach().cpu().numpy()
    expected_position = np.asarray(
        cfg.blcs.expected_selected_position_m,
        dtype=np.float64,
    )
    if not np.allclose(actual_position, expected_position, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            "Frozen BLCS pilot selected position changed: "
            f"expected {expected_position.tolist()}, got {actual_position.tolist()}."
        )

    static_scene_path = _verified_file(
        cfg.static_scene.path,
        expected_sha256=str(cfg.static_scene.sha256),
    )
    with np.load(static_scene_path, allow_pickle=False) as payload:
        if set(payload.files) != {"rgb", "depth", "alpha"}:
            raise ValueError("Static B00 scene payload keys changed.")
        static_rgb = np.asarray(payload["rgb"])
        static_depth = np.asarray(payload["depth"])
    camera_id = str(cfg.pilot.camera_id)
    camera = next(
        (camera for camera in contract.cameras if camera.camera_id == camera_id),
        None,
    )
    if camera is None:
        raise ValueError(f"Pilot camera {camera_id!r} is absent.")
    renderer = DeterministicCpuSphereRenderer(
        scene_fingerprint=contract.scene_fingerprint,
        frames={
            camera_id: CpuSceneFrame(
                rgb=static_rgb,
                depth=static_depth,
            )
        },
    )
    output_dir = publish_single_frame_pilot(
        scene=scene,
        scene_contract=contract,
        renderer=renderer,
        config=SingleFramePilotConfig(
            camera_id=camera_id,
            trajectory_frame_index=selected_index,
            ball_radius_m=float(cfg.pilot.ball_radius_m),
            ball_color_rgb=cast(
                tuple[int, int, int],
                tuple(int(value) for value in cfg.pilot.ball_color_rgb),
            ),
            supersampling=int(cfg.pilot.supersampling),
            jpeg_quality=int(cfg.pilot.jpeg_quality),
        ),
        provenance=PilotProvenance(
            seed=seed,
            scene_contract_uri=_relative_uri(contract_path, repo_root),
            scene_contract_sha256=str(cfg.scene_contract.sha256),
            static_scene_uri=_relative_uri(static_scene_path, repo_root),
            static_scene_sha256=str(cfg.static_scene.sha256),
            static_scene_request_fingerprint=str(cfg.static_scene.request_fingerprint),
            git_revision=_git(repo_root, "rev-parse", "HEAD"),
            git_dirty=bool(_git(repo_root, "status", "--porcelain=v1")),
            code_diff_sha256=_code_identity(repo_root),
        ),
        output_root=_path(cfg.output_root),
    )
    manifest = load_and_validate_single_frame_pilot(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "scene_id": scene.scene_id,
                "trajectory_frame_count": int(scene.ball_pos_world.shape[0]),
                "trajectory_frame_index": selected_index,
                "court_position_m": actual_position.tolist(),
                "positive": manifest["frames"][0],
                "negative": manifest["frames"][1],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verified_file(value: Any, *, expected_sha256: str) -> Path:
    path = _path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Artifact SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}."
        )
    return path


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _relative_uri(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_identity(repo_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git(repo_root, "status", "--porcelain=v1").encode())
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    )
    for relative in (
        "src/synthetic_data_generation/rendering/cpu_fake_renderer.py",
        "src/synthetic_data_generation/rendering/renderer_port.py",
        "src/synthetic_data_generation/dataset/single_frame_pilot.py",
        "src/synthetic_data_generation/scripts/publish_b00_ball_pilot.py",
        "src/synthetic_data_generation/configs/publish_b00_ball_pilot.yaml",
    ):
        path = repo_root / relative
        digest.update(relative.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
