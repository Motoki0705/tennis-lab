"""Immutable, reproducible motion-disjoint PLCS dataset versions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.utils.checksum import dual_sha256 as file_sha256
from src.utils.io import save_json_atomic

_SPLITS = ("train", "val", "test")
_OUTPUTS = (
    "meta.json",
    "scenes_meta.json",
    "train.txt",
    "val.txt",
    "test.txt",
    "split_info.json",
    "config.yaml",
)


def _records(source: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    meta = json.loads((source / "meta.json").read_text())
    rows = json.loads((source / "scenes_meta.json").read_text())
    inventory: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["scene_id"]
        if (
            type(name) is not str
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("scene_id must be a single path component")
        if (
            name in inventory
            or type(row.get("motion_source")) is not str
            or not row["motion_source"].strip()
        ):
            raise ValueError(
                "Require unique scene IDs and explicit nonempty motion sources"
            )
        inventory[name] = row
    names = [row["scene_id"] for row in meta["scenes"]]
    if len(names) != len(set(names)) or set(names) != set(inventory):
        raise ValueError("Scene inventory and motion-source metadata disagree")
    for row in meta["scenes"]:
        if row["file"] != row["scene_id"]:
            raise ValueError("Scene file must equal its explicit scene ID")
        if not (source / "scenes" / row["scene_id"]).is_dir():
            raise ValueError(f"Missing scene directory: {row['scene_id']}")
    return meta, inventory


def _paths(source: Path, destination: Path, seed: int) -> tuple[Path, Path]:
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("source and destination must be explicit absolute paths")
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Use separate dataset version directories")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return source, destination


def _existing(destination: Path, identity: dict[str, Any]) -> bool:
    if not destination.exists():
        return False
    receipt = destination / "dataset_version.json"
    if not receipt.is_file():
        raise ValueError("Destination is incomplete or has another recipe")
    saved = json.loads(receipt.read_text())
    if saved.get("recipe") != identity or saved.get("outputs") != {
        name: file_sha256(destination / name) for name in _OUTPUTS
    }:
        raise ValueError("Destination is incomplete, modified or has another recipe")
    _, records = _records(destination)
    _validate_splits(destination, records)
    return True


def _publish(staging: Path, destination: Path, identity: dict[str, Any]) -> None:
    save_json_atomic(
        {
            "recipe": identity,
            "outputs": {name: file_sha256(staging / name) for name in _OUTPUTS},
        },
        staging / "dataset_version.json",
    )
    staging.rename(destination)


def _validate_splits(
    source: Path, records: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    info = json.loads((source / "split_info.json").read_text())
    if info.get("split_group") != "motion_source":
        raise ValueError("Parent dataset must use motion-disjoint splits")
    all_names: set[str] = set()
    owners: dict[str, str] = {}
    result: dict[str, list[str]] = {}
    for split in _SPLITS:
        names = (source / f"{split}.txt").read_text().splitlines()
        if (
            not names
            or len(names) != len(set(names))
            or set(names) & all_names
            or set(names) - records.keys()
        ):
            raise ValueError(
                "Parent split has empty, duplicate, overlapping or unknown scenes"
            )
        all_names.update(names)
        for name in names:
            motion = records[name]["motion_source"]
            if motion in owners and owners[motion] != split:
                raise ValueError("Parent motion source leaks across splits")
            owners[motion] = split
            if info["source_assignments"].get(motion) != split:
                raise ValueError("Parent source assignment disagrees with split files")
        result[split] = sorted(names)
    if all_names != records.keys():
        raise ValueError("Parent splits must cover the complete scene inventory")
    return result


def prepare_motion_split(source: Path, destination: Path, *, seed: int) -> None:
    source, destination = _paths(source, destination, seed)
    meta, records = _records(source)
    identity = {
        "source": str(source),
        "source_meta_sha256": file_sha256(source / "meta.json"),
        "source_scenes_meta_sha256": file_sha256(source / "scenes_meta.json"),
        "source_config_sha256": file_sha256(source / "config.yaml"),
        "split_group": "motion_source",
        "seed": seed,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    if _existing(destination, identity):
        return
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise ValueError(f"Incomplete previous split preparation: {staging}")
    writer = PLCSDatasetWriter(
        staging, court_keypoint_contract=resolve_court_keypoint_contract("physical_v1")
    )
    for entry in source.iterdir():
        if entry.name in {
            "train.txt",
            "val.txt",
            "test.txt",
            "split_info.json",
            "dataset_version.json",
        }:
            continue
        target = staging / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, copy_function=os.link, dirs_exist_ok=True)
        else:
            shutil.copyfile(entry, target)
    writer.scene_records = [
        {**row, "motion_source": records[row["scene_id"]]["motion_source"]}
        for row in meta["scenes"]
    ]
    writer.save_motion_group_splits(val_ratio=0.1, test_ratio=0.1, seed=seed)
    _validate_splits(staging, records)
    _publish(staging, destination, identity)


def prepare_subset(
    source: Path, destination: Path, *, seed: int, train: int, evaluation: int
) -> None:
    source, destination = _paths(source, destination, seed)
    if any(type(count) is not int or count <= 0 for count in (train, evaluation)):
        raise ValueError("Subset counts must be positive integers")
    meta, records = _records(source)
    splits = _validate_splits(source, records)
    identity = {
        "source": str(source),
        "seed": seed,
        "train": train,
        "evaluation": evaluation,
        "inputs": {name: file_sha256(source / name) for name in _OUTPUTS},
    }
    if _existing(destination, identity):
        return
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[str]] = {}
    for split, count in (("train", train), ("val", evaluation), ("test", evaluation)):
        if count > len(splits[split]):
            raise ValueError(
                f"Invalid {split} subset size {count}/{len(splits[split])}"
            )
        chosen[split] = sorted(rng.choice(splits[split], count, replace=False).tolist())
    inventory = set().union(*map(set, chosen.values()))
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise ValueError(f"Incomplete previous subset preparation: {staging}")
    staging.mkdir(parents=True)
    (staging / "scenes").mkdir()
    for name in sorted(inventory):
        shutil.copytree(
            source / "scenes" / name, staging / "scenes" / name, copy_function=os.link
        )
    meta["scenes"] = [row for row in meta["scenes"] if row["scene_id"] in inventory]
    meta["stats"] = {"subset": True, "num_scenes": len(inventory)}
    save_json_atomic(meta, staging / "meta.json")
    save_json_atomic(
        [row for name, row in records.items() if name in inventory],
        staging / "scenes_meta.json",
    )
    shutil.copyfile(source / "config.yaml", staging / "config.yaml")
    for split, names in chosen.items():
        (staging / f"{split}.txt").write_text("\n".join(names) + "\n")
    parent = json.loads((source / "split_info.json").read_text())
    save_json_atomic(
        {
            **parent,
            "n_scenes": {s: len(n) for s, n in chosen.items()},
            "subset_seed": seed,
        },
        staging / "split_info.json",
    )
    _publish(staging, destination, identity)
