"""Generate one shared dataset through the BLCS physics adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_refiner_3d.configuration.generation import (
    CameraSampling,
    GenerationConfig,
    generation_config,
)
from src.tasks.ball_refiner_3d.data.schema import SCHEMA
from src.tasks.ball_refiner_3d.generate_dataset.blcs_adapter import simulate_rally
from src.tasks.ball_refiner_3d.generate_dataset.writer import write_rally
from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.utils.io import write_json_atomic
from src.utils.physics.ball.record import EVENT_BITS, RECORD_SCHEMA


def _generate_rally(
    index: int,
    root: Path,
    generation: GenerationConfig,
    camera: CameraSampling,
    physics: GeneratorConfig,
    split: tuple[str, ...],
) -> dict[str, Any]:
    result, record, xyz, cameras, attempts = simulate_rally(
        index, generation, camera, physics
    )
    return write_rally(
        root,
        index,
        generation,
        camera,
        split[index],
        result,
        record,
        xyz,
        cameras,
        attempts,
    )


def generate_dataset(config: DictConfig) -> Path:
    raw, root, generation, camera, physics = generation_config(config)
    if root.exists():
        raise FileExistsError(f"Dataset output already exists: {root}")
    (root / "rallies").mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(raw), root / "config.yaml")
    write_json_atomic(root / "state.json", {"status": "generating"})
    splits = np.array(
        ["train"] * generation.train_rallies
        + ["val"] * generation.val_rallies
        + ["test"] * generation.test_rallies
    )
    np.random.default_rng(generation.seed).shuffle(splits)
    start = time.monotonic()
    records = []
    args = (root, generation, camera, physics, tuple(str(s) for s in splits))
    indices = list(range(len(splits)))
    iterator = (
        map(lambda i: _generate_rally(i, *args), indices)
        if generation.workers == 1
        else run_parallel_scene_generation(
            _generate_rally, indices, *args, num_workers=generation.workers
        )
    )
    for record in iterator:
        records.append(record)
        if len(records) % 25 == 0 or len(records) == len(indices):
            print(
                json.dumps(
                    {
                        "generated": len(records),
                        "total": len(indices),
                        "seconds": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
    manifest = {
        "schema": SCHEMA,
        "fps": physics.rally.output_fps,
        "image_size_wh": [camera.width, camera.height],
        "views": camera.views,
        "event_bits": EVENT_BITS,
        "physics_record": RECORD_SCHEMA,
        "records": records,
        "split_policy": "seeded rally-disjoint; every view and temporal window follows its parent rally",
        "selection_policy": "truncate at first fence exit; bounded resampling of short/no-full-rally-visible-camera/full-physics-rejected rallies; attempts in each record",
        "event_policy": "events of the physics record at their first output frame at or after the simulation step; toss excluded; post-return bounces never simulated into the trajectory",
        "surface_policy": "one court surface per rally, drawn uniformly from physics.surface_choices",
        "camera_policy": "fixed within rally; continuous X/Z on both baseline-rear fence planes; clean trajectory fully in frame before occlusion/noise",
        "generation_seconds": time.monotonic() - start,
    }
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(
        root / "state.json", {"status": "complete", "rallies": len(records)}
    )
    print(
        json.dumps(
            {
                "dataset": str(root),
                "rallies": len(records),
                "frames": sum(r["frames"] for r in records),
            }
        ),
        flush=True,
    )
    return Path(root)
