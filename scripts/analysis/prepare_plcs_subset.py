"""Freeze a seeded subset while preserving every parent motion-source split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.tennis_scene.generate_dataset.manifest import file_sha256
from src.utils.io import save_json_atomic


def prepare(
    source: Path, destination: Path, *, seed: int, train: int, evaluation: int
) -> None:
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Use separate dataset version directories")
    identity = {
        "source": str(source),
        "seed": seed,
        "train": train,
        "evaluation": evaluation,
        "inputs": {
            name: file_sha256(source / name)
            for name in (
                "meta.json",
                "scenes_meta.json",
                "train.txt",
                "val.txt",
                "test.txt",
                "split_info.json",
            )
        },
    }
    receipt = destination / "dataset_version.json"
    if destination.exists():
        if receipt.is_file() and json.loads(receipt.read_text()) == identity:
            print(f"Using subset {destination}")
            return
        raise ValueError("Subset destination has another recipe or is incomplete")
    parent_split = json.loads((source / "split_info.json").read_text())
    if parent_split.get("split_group") != "motion_source":
        raise ValueError("Parent dataset must already use motion-disjoint splits")
    rng = np.random.default_rng(seed)
    chosen = {}
    for split, count in (("train", train), ("val", evaluation), ("test", evaluation)):
        names = sorted((source / f"{split}.txt").read_text().splitlines())
        if not 0 < count <= len(names):
            raise ValueError(f"Invalid {split} subset size {count}/{len(names)}")
        chosen[split] = sorted(rng.choice(names, count, replace=False).tolist())
    inventory = set().union(*map(set, chosen.values()))
    if len(inventory) != train + 2 * evaluation:
        raise ValueError("Parent splits overlap")
    staging = destination.with_name(destination.name + ".building")
    staging.mkdir(parents=True)
    (staging / "scenes").mkdir()
    for name in sorted(inventory):
        shutil.copytree(
            source / "scenes" / name, staging / "scenes" / name, copy_function=os.link
        )
    meta = json.loads((source / "meta.json").read_text())
    meta["scenes"] = [row for row in meta["scenes"] if row["scene_id"] in inventory]
    meta["stats"] = {"subset": True, "num_scenes": len(inventory)}
    records = [
        row
        for row in json.loads((source / "scenes_meta.json").read_text())
        if row["scene_id"] in inventory
    ]
    save_json_atomic(meta, staging / "meta.json")
    save_json_atomic(records, staging / "scenes_meta.json")
    shutil.copyfile(source / "config.yaml", staging / "config.yaml")
    for split, names in chosen.items():
        (staging / f"{split}.txt").write_text("\n".join(names) + "\n")
    save_json_atomic(
        {
            **parent_split,
            "n_scenes": {s: len(n) for s, n in chosen.items()},
            "subset_seed": seed,
        },
        staging / "split_info.json",
    )
    save_json_atomic(identity, staging / receipt.name)
    staging.rename(destination)
    print({split: len(names) for split, names in chosen.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--evaluation", type=int, default=200)
    args = parser.parse_args()
    prepare(
        args.source,
        args.destination,
        seed=args.seed,
        train=args.train,
        evaluation=args.evaluation,
    )
