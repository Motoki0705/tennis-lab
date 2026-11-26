"""Generate one shared dataset through the BLCS physics adapter."""

from __future__ import annotations

import json
import time
from collections import Counter
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
from src.tasks.blcs.generate_dataset.simulation.net_contact import NetContact
from src.utils.io import write_json_atomic
from src.utils.physics.ball.record import EVENT_BITS, EVENT_KINDS, RECORD_SCHEMA


def _generate_rally(
    index: int,
    root: Path,
    generation: GenerationConfig,
    camera: CameraSampling,
    physics: GeneratorConfig,
    split: tuple[str, ...],
    net_contacts: dict[int, NetContact],
) -> dict[str, Any]:
    net_contact = net_contacts.get(index)
    result, record, xyz, cameras, attempts = simulate_rally(
        index, generation, camera, physics, net_contact
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
        net_contact,
    )


def assign_net_contacts(
    splits: np.ndarray, fraction: float, seed: int
) -> dict[int, NetContact]:
    """Reserve an exact, balanced quota within each rally-disjoint split."""
    rng = np.random.default_rng(seed)
    requests: dict[int, NetContact] = {}
    for split in ("train", "val", "test"):
        rows = np.flatnonzero(splits == split)
        count = int(len(rows) * fraction + 0.5)
        for index, row in enumerate(rng.permutation(rows)[:count]):
            requests[int(row)] = (
                NetContact.CROSSED if index % 2 == 0 else NetContact.DROPPED
            )
    return requests


def _event_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    events: Counter[str] = Counter()
    contacts: Counter[str] = Counter()
    for record in records:
        events.update(record["event_counts"])
        if record["net_contact"] is not None:
            contacts.update([record["net_contact"]])
    return {
        "rallies": len(records),
        "event_counts": {kind: events[kind] for kind in EVENT_KINDS},
        "net_contact_counts": {kind.value: contacts[kind.value] for kind in NetContact},
        "net_contact_fraction": sum(
            record["event_counts"]["net"] > 0 for record in records
        )
        / len(records),
    }


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
    net_contacts = assign_net_contacts(
        splits, generation.net_contact_fraction, generation.seed
    )
    start = time.monotonic()
    records = []
    args = (
        root,
        generation,
        camera,
        physics,
        tuple(str(s) for s in splits),
        net_contacts,
    )
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
        "physics_verification": "float32 bit equality"
        if generation.net_contact_fraction > 0
        else "max error <= 1e-6 m",
        "requested_net_contact_fraction": generation.net_contact_fraction,
        **_event_summary(records),
        "events_by_split": {
            split: _event_summary(
                [record for record in records if record["split"] == split]
            )
            for split in ("train", "val", "test")
        },
        "records": records,
        "split_policy": "seeded rally-disjoint; every view and temporal window follows its parent rally",
        "selection_policy": "truncate at first fence exit; bounded resampling of short/no-full-rally-visible-camera/full-physics-rejected rallies; attempts in each record",
        "event_policy": "counts include all recorded physical events; output labels use the first frame at or after the simulation step, excluding toss",
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
