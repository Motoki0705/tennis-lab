"""Reproject an existing shared dataset without resimulating any 3D trajectory."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tasks.ball_refiner_3d.configuration.generation import CameraSampling
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.generate_dataset.cameras import sample_visible_cameras
from src.utils.io import write_json_atomic


def reproject_dataset(
    source: Path, output: Path, config: CameraSampling, *, seed: int
) -> Path:
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("Reprojection requires a new output directory")
    dataset = SharedDataset(source)
    if (config.width, config.height) != dataset.image_size:
        raise ValueError("Reprojection must retain the reference image grid")
    (output / "rallies").mkdir(parents=True)
    write_json_atomic(output / "state.json", {"status": "reprojecting"})
    start = time.monotonic()
    manifest = dataset.manifest
    for rally, record in zip(dataset.rallies, manifest["records"], strict=True):
        cameras = sample_visible_cameras(
            config, rally.xyz, np.random.default_rng(seed + rally.index)
        )
        if cameras is None:
            raise RuntimeError(
                f"No full-rally-visible cameras within budget for {rally.name}; 3D truth was not changed"
            )
        with np.load(source / record["path"], allow_pickle=False) as values:
            arrays = {key: values[key] for key in values.files}
        xyz_hash = hashlib.sha256(arrays["xyz_m"].tobytes()).hexdigest()
        arrays.update(
            {
                "uv_px": np.stack(
                    [camera.project(rally.xyz)[0] for camera in cameras]
                ).astype(np.float32),
                "visible": np.ones((config.views, len(rally.xyz)), dtype=bool),
                "projection": np.stack([camera.matrix for camera in cameras]),
                "camera_centers": np.stack([camera.center for camera in cameras]),
                "intrinsic": np.stack([camera.intrinsic for camera in cameras]),
                "rotation": np.stack([camera.rotation for camera in cameras]),
                "translation": np.stack([camera.translation for camera in cameras]),
            }
        )
        path = output / record["path"]
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(path, allow_pickle=False) as saved:
            if hashlib.sha256(saved["xyz_m"].tobytes()).hexdigest() != xyz_hash:
                raise AssertionError("Reprojection changed 3D ground truth")
        record.update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            xyz_sha256=xyz_hash,
            visible_fraction=1.0,
        )
        write_json_atomic(path.with_suffix(".json"), record)
        if (rally.index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "reprojected": rally.index + 1,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    manifest.update(
        views=config.views,
        camera_policy="fixed rear-fence cameras; clean full trajectory must be in frame; no coordinate clipping",
        reprojection={
            "source_manifest_sha256": dataset.manifest_hash,
            "seed": seed,
            "camera_sampling": asdict(config),
            "preserved": "byte-identical xyz, event, time arrays and rally splits",
            "seconds": time.monotonic() - start,
        },
    )
    raw = OmegaConf.load(source / "config.yaml")
    raw.camera_sampling = OmegaConf.create(asdict(config))
    OmegaConf.save(raw, output / "config.yaml")
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(
        output / "state.json", {"status": "complete", "rallies": len(dataset.rallies)}
    )
    return output
