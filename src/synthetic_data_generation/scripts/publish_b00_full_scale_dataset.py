"""
Publish the frozen full-scale BLCS × B00 3DGS TrackNet training dataset.

Usage:
    python -m src.synthetic_data_generation.scripts.publish_b00_full_scale_dataset

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/publish_b00_full_scale_dataset.yaml`.
    - Every BLCS trajectory is independently seeded in court metres.
    - The accepted SceneContract applies Sim(3) exactly once per trajectory.
    - The synthetic publisher has no dependency on ball-detection internals.
    - This orchestration script runs a narrow TrackNet DataLoader smoke check
      only after the fingerprinted dataset has been atomically published.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.synthetic_data_generation.code_identity import compute_code_identity
from src.synthetic_data_generation.dataset.full_scale_dataset import (
    FullScaleDatasetConfig,
    FullScaleProvenance,
    StaticSceneProvenance,
    TrajectorySamplingSpec,
    load_and_validate_full_scale_dataset,
    publish_full_scale_dataset,
)
from src.synthetic_data_generation.rendering.cpu_fake_renderer import (
    CpuSceneFrame,
    DeterministicCpuSphereRenderer,
)
from src.synthetic_data_generation.rendering.gsplat_subprocess_adapter import (
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneGenerator
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="publish_b00_full_scale_dataset",
)
def main(cfg: DictConfig) -> int:
    """Generate frozen rallies, publish grouped clips, and smoke the consumer."""
    repo_root = Path(to_absolute_path(".")).resolve()
    contract_path = _verified_file(
        cfg.scene_contract.path,
        expected_sha256=str(cfg.scene_contract.sha256),
    )
    contract = load_scene_contract(contract_path)
    if contract.scene_fingerprint != str(cfg.scene_contract.scene_fingerprint):
        raise ValueError("Full-scale SceneContract fingerprint mismatch.")

    static_frames: dict[str, CpuSceneFrame] = {}
    static_provenance: list[StaticSceneProvenance] = []
    camera_ids: list[str] = []
    for source in cfg.static_scenes:
        camera_id = str(source.camera_id)
        if camera_id in static_frames:
            raise ValueError(f"Duplicate full-scale camera: {camera_id}")
        path = _verified_file(source.path, expected_sha256=str(source.sha256))
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"rgb", "depth", "alpha"}:
                raise ValueError(f"Static B00 payload keys changed: {path}")
            static_frames[camera_id] = CpuSceneFrame(
                rgb=np.asarray(payload["rgb"]),
                depth=np.asarray(payload["depth"]),
            )
        camera_ids.append(camera_id)
        static_provenance.append(
            StaticSceneProvenance(
                camera_id=camera_id,
                uri=_relative_uri(path, repo_root),
                sha256=str(source.sha256),
                request_fingerprint=str(source.request_fingerprint),
            )
        )

    seed_base = int(cfg.trajectory.seed_base)
    count = int(cfg.trajectory.count)
    specs = tuple(
        TrajectorySamplingSpec(
            seed=seed_base + index,
            from_cell=index % 9,
            side="near" if index % 2 == 0 else "far",
            scene_id=f"b00-blcs-full-{index:03d}",
        )
        for index in range(count)
    )
    scenes = []
    generator = BLCSSceneGenerator(device="cpu")
    for index, spec in enumerate(specs, start=1):
        _seed_everything(spec.seed)
        scene = generator.generate_scene(
            spec.from_cell,
            spec.side,
            spec.scene_id,
        )
        if scene is None:
            raise RuntimeError(f"BLCS generation returned no scene: {spec.scene_id}")
        scenes.append(scene)
        print(
            f"generated trajectory {index}/{count}: "
            f"{scene.scene_id} frames={int(scene.ball_pos_world.shape[0])}",
            flush=True,
        )

    renderer = DeterministicCpuSphereRenderer(
        scene_fingerprint=contract.scene_fingerprint,
        frames=static_frames,
    )
    last_reported = 0

    def report_progress(done: int, total: int) -> None:
        nonlocal last_reported
        if done == total or done - last_reported >= int(cfg.progress_interval_frames):
            print(f"rendered frames {done}/{total}", flush=True)
            last_reported = done

    output_dir = publish_full_scale_dataset(
        scenes=scenes,
        scene_contract=contract,
        renderer=renderer,
        config=FullScaleDatasetConfig(
            camera_ids=tuple(camera_ids),
            trajectories=specs,
            clip_length=int(cfg.dataset.clip_length),
            ball_radius_m=float(cfg.dataset.ball_radius_m),
            ball_color_rgb=cast(
                tuple[int, int, int],
                tuple(int(value) for value in cfg.dataset.ball_color_rgb),
            ),
            supersampling=int(cfg.dataset.supersampling),
            jpeg_quality=int(cfg.dataset.jpeg_quality),
        ),
        provenance=FullScaleProvenance(
            scene_contract_uri=_relative_uri(contract_path, repo_root),
            scene_contract_sha256=str(cfg.scene_contract.sha256),
            static_scenes=tuple(static_provenance),
            git_revision=_git(repo_root, "rev-parse", "HEAD"),
            git_dirty=bool(_git(repo_root, "status", "--porcelain=v1")),
            code_diff_sha256=compute_code_identity(repo_root),
        ),
        output_root=_path(cfg.output_root),
        progress=report_progress,
    )
    manifest = load_and_validate_full_scale_dataset(output_dir)
    smoke = _tracknet_dataloader_smoke(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "label_statistics": manifest["label_statistics"],
                "diversity": manifest["diversity"],
                "dataloader_smoke": smoke,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _tracknet_dataloader_smoke(dataset_dir: Path) -> dict[str, Any]:
    config = OmegaConf.create(
        {
            "model": {"num_frames": 8},
            "data": {
                "data_dir": str(dataset_dir),
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "sample_stride": 64,
                "image_size": [288, 512],
                "heatmap_size": [144, 256],
                "split": {"train_file": "splits/train.txt"},
            },
        }
    )
    data_module = TrackNetDataModule(config)
    dataset = data_module.create_dataset(
        split_name="train",
        split_file="splits/train.txt",
        augmentation=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    return {
        "window_count": len(dataset),
        "batch_images_shape": list(batch["images"].shape),
        "batch_heatmaps_shape": list(batch["heatmaps"].shape),
        "batch_visible_instances": float(batch["visibility"].sum()),
        "batch_heatmap_max": float(batch["heatmaps"].max()),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


if __name__ == "__main__":
    raise SystemExit(main())
