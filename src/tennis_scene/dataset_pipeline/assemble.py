"""Publish a complete, versioned multi-source SLCS dataset and fixed splits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import exact_config_mapping
from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tasks.slcs.data.dino_tokens import load_dino_spec, load_dino_tokens
from src.tasks.slcs.data.splits import save_split_file
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifest,
    load_dataset_manifest,
    validate_id_component,
)
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.io import save_json_atomic
from src.utils.paths import PROJECT_ROOT


def validate_assembly_config(cfg: DictConfig) -> None:
    exact_config_mapping(
        cfg,
        path="assembly",
        required_keys={
            "paths",
            "source_datasets",
            "dataset_directory",
            "dataset_id",
            "clip_ids",
            "seed",
            "video_splits",
            "output_dir",
        },
    )
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("assembly.seed must be a nonnegative integer")
    validate_id_component(cfg.dataset_id, field_name="dataset_id")
    if not cfg.source_datasets or any(
        type(path) is not str for path in cfg.source_datasets
    ):
        raise ValueError("source_datasets must contain data-root-relative paths")
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )
    for source in cfg.source_datasets:
        resolver.resolve(PathRole.DATA, source)
    resolver.resolve(PathRole.DATA, cfg.dataset_directory)
    resolver.resolve(PathRole.OUTPUT, cfg.output_dir)


def assemble_dataset(
    sources: list[Path],
    destination: Path,
    *,
    dataset_id: str,
    video_splits: dict[str, str],
    seed: int,
    clip_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Rebind only dataset identity; validate arrays and preserve original receipts.

    Media, feature and scene arrays are hard-linked. A new clip manifest and
    completion markers record the dataset-id-only change and source hashes.
    Split assignment is explicit at recording/venue level, never window-random.
    """
    validate_id_component(dataset_id, field_name="dataset_id")
    sources = [path.resolve() for path in sources]
    destination = destination.resolve()
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Select unique nonempty source datasets")
    if any(
        destination == path
        or destination.is_relative_to(path)
        or path.is_relative_to(destination)
        for path in sources
    ):
        raise ValueError("Assembly destination must be independent of every input")
    records = {}
    for source in sources:
        for key, record in load_dataset_manifest(source).clips.items():
            if key in records:
                raise ValueError(f"Duplicate clip ID across datasets: {key}")
            records[key] = (source, record)
    if clip_ids is not None:
        if (
            not clip_ids
            or len(set(clip_ids)) != len(clip_ids)
            or set(clip_ids) - records.keys()
        ):
            raise ValueError("Assembly clip_ids must be unique existing clips")
        records = {key: records[key] for key in sorted(clip_ids)}
    videos = {record.video_id for _, record in records.values()}
    if set(video_splits) != videos or set(video_splits.values()) != {
        "train",
        "val",
        "test",
    }:
        raise ValueError(
            "Explicit split assignments must cover exactly the selected videos and all three splits"
        )
    identity: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "seed": seed,
        "video_splits": video_splits,
        "clips": {},
    }
    token_identity = None
    for key, (source, record) in records.items():
        clip = ClipManifest.load(source / record.path)
        load_slcs_annotation(clip)
        spec = load_dino_spec(clip.clip_dir)
        token_marker = json.loads(
            (clip.clip_dir / "annotations/dino_v3/annotation.json").read_text()
        )
        current_token_identity = (
            spec,
            token_marker["generator"].get("checkpoint_sha256"),
        )
        checkpoint_digest = current_token_identity[1]
        if (
            not isinstance(checkpoint_digest, str)
            or len(checkpoint_digest) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_digest)
        ):
            raise ValueError(
                "Feature receipts must identify the checkpoint with SHA-256"
            )
        if token_identity is not None and current_token_identity != token_identity:
            raise ValueError(
                "All sources must use the same RGB feature spec and checkpoint"
            )
        token_identity = current_token_identity
        for camera in clip.camera_ids:
            load_dino_tokens(clip, camera, expected_spec=spec)
        hashes = {
            str(path.relative_to(clip.clip_dir)): sha256(path)
            for path in sorted((clip.clip_dir / "annotations").rglob("*"))
            if path.is_file()
        }
        hashes.update({name: sha256(clip.clip_dir / name) for name in clip.video_paths})
        hashes["clip.json"] = clip.digest()
        identity["clips"][key] = {
            "source": str(source),
            "path": record.path,
            "sha256": hashes,
        }
    if token_identity is None:
        raise ValueError("Assembly requires at least one complete source clip")
    receipt = destination / "assembly.json"
    if destination.exists():
        if (
            not receipt.exists()
            or json.loads(receipt.read_text())["inputs"] != identity
        ):
            raise ValueError(
                "Assembly inputs changed or destination is incomplete; choose a new dataset version"
            )
        saved = json.loads(receipt.read_text())
        target_index = load_dataset_manifest(destination)
        if target_index.dataset_id != dataset_id or target_index.clips != {
            key: record for key, (_, record) in records.items()
        }:
            raise ValueError("Assembled dataset inventory changed")
        if sha256(destination / "splits.json") != saved["split_sha256"]:
            raise ValueError("Assembled split assignments changed")
        for key, (_, record) in records.items():
            clip = ClipManifest.load(destination / record.path)
            if (
                clip.digest()
                != json.loads(receipt.read_text())["derived_manifest_sha256"][key]
            ):
                raise ValueError(f"Assembled manifest changed: {key}")
            load_slcs_annotation(clip)
            feature_receipt = json.loads(
                (clip.clip_dir / "annotations/dino_v3/annotation.json").read_text()
            )
            if (
                load_dino_spec(clip.clip_dir),
                feature_receipt["generator"].get("checkpoint_sha256"),
            ) != token_identity:
                raise ValueError(f"Assembled feature identity changed: {key}")
            for camera in clip.camera_ids:
                load_dino_tokens(clip, camera, expected_spec=token_identity[0])
        return cast(dict[str, Any], json.loads(receipt.read_text()))
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete assembly transaction requires inspection: {temporary}"
        )
    temporary.mkdir(parents=True)
    index = DatasetManifest(dataset_id)
    derived = {}
    for key, (source, record) in records.items():
        original = source / record.path
        target = temporary / record.path
        target.mkdir(parents=True)
        raw = json.loads((original / "clip.json").read_text())
        raw["dataset_id"] = dataset_id
        save_json_atomic(raw, target / "clip.json")
        new_digest = sha256(target / "clip.json")
        derived[key] = new_digest
        original_digest = identity["clips"][key]["sha256"]["clip.json"]
        for name in identity["clips"][key]["sha256"]:
            if name == "clip.json":
                continue
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in {
                "annotations/tennis_scene/annotation.json",
                "annotations/dino_v3/annotation.json",
            }:
                marker = json.loads((original / name).read_text())
                digest_field = (
                    "clip_manifest_sha256"
                    if "tennis_scene" in name
                    else "input_manifest_digest"
                )
                if marker[digest_field] != original_digest:
                    raise ValueError(f"Source marker changed during assembly: {key}")
                marker[digest_field] = new_digest
                marker["derived_from"] = {
                    "manifest_sha256": original_digest,
                    "dataset": str(source),
                    "change": "dataset_id only",
                }
                save_json_atomic(marker, path)
            else:
                os.link(original / name, path)
        index.clips[key] = record
    index.save(temporary)
    counts = {
        split: sum(1 for selected in video_splits.values() if selected == split)
        for split in ("train", "val", "test")
    }
    save_split_file(
        temporary / "splits.json",
        video_splits,
        seed=seed,
        val_ratio=counts["val"] / len(videos),
        test_ratio=counts["test"] / len(videos),
    )
    result = {
        "inputs": identity,
        "derived_manifest_sha256": derived,
        "split_unit": "source recording / curated broadcast venue",
        "num_clips": len(records),
        "split_sha256": sha256(temporary / "splits.json"),
    }
    save_json_atomic(result, temporary / "assembly.json")
    temporary.rename(destination)
    return result
