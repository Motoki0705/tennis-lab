"""Freeze all-frame manifests while leaving the existing snapshot untouched."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    build_all_frame_manifests,
    validate_all_frame_manifest,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    common_intervals,
)
from src.tasks.ball_detection.data.windows.manifest import (
    COORDINATE_MANIFEST_SCHEMA,
    clip_semantic_digest,
)
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version


def _reusable_shard_hashes(store: BallFrameStore, manifest: Path | None) -> dict[str, str]:
    """Only an explicitly supplied coordinate manifest of this exact snapshot is reused."""
    if manifest is None:
        return {}
    raw = json.loads(manifest.read_text())
    identity = {name: dual_sha256(store.directory / name) for name in ("metadata.json", "index.npz")}
    if (raw["schema"] != COORDINATE_MANIFEST_SCHEMA or raw["input_kind"] != "mdd_only"
            or Path(raw["ball_store"]["directory"]).resolve() != store.directory.resolve()
            or raw["ball_store"]["hashes"] != identity):
        raise ValueError("RGB hash source must be an MDD-only coordinate manifest of the identical snapshot")
    hashes = {}
    for record in raw["clips"]:
        clip_id = record["clip"]["clip_id"]
        clip = store.clip_by_id(clip_id)
        if clip_id in hashes or record["semantic_sha256"] != clip_semantic_digest(store, clip):
            raise ValueError("RGB hash source contains duplicate or changed clip identities")
        hashes[clip_id] = record["rgb_shard_sha256"]
    return hashes


def prepare_all_frame_training(
    ball_directory: Path, output: Path, *, configs: Sequence[AllFrameWindowConfig],
    rgb_manifest: Path | None = None,
    external_payloads: Mapping[str, Path] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Reference and hash the source snapshot; never create or remove RGB links.

    The existing snapshot must remain immutable and available for the lifetime
    of the prepared data. Distribution to an independent store is a separate API.
    """
    if not ball_directory.is_absolute() or not output.is_absolute() or (rgb_manifest is not None and not rgb_manifest.is_absolute()):
        raise ValueError("All-frame snapshot, output and optional RGB manifest paths must be absolute")
    ball_directory, output = ball_directory.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"All-frame preparation requires a new output directory: {output}")
    if not configs:
        raise ValueError("All-frame preparation requires at least one T")
    if output.is_relative_to(ball_directory) or ball_directory.is_relative_to(output):
        raise ValueError("Preparation output must be separate from the source snapshot")
    store = BallFrameStore(ball_directory, external_payloads=external_payloads)
    code = source_identity()
    paths = {ball_directory / "index.npz", ball_directory / "metadata.json"}
    paths.update(store.image_path(clip) for clip in store.clips)
    for clip in store.clips:
        if clip.source != "chat_annotation":
            continue
        rows = store.clip_rows(clip)
        has_empty = np.any(store.frames["annotated"][rows] & store.frames["is_target"][rows]
                           & (store.frames["inst_count"][rows] == 0))
        if has_empty and common_intervals(annotation_states(store, clip), configs[0]):
            paths.add(store.annotation_path(clip))
    if rgb_manifest is not None:
        paths.add(rgb_manifest)
    versions = {path: file_version(path) for path in paths}
    hashes = _reusable_shard_hashes(store, rgb_manifest)
    for i, clip in enumerate(store.clips):
        if clip.clip_id not in hashes:
            hashes[clip.clip_id] = dual_sha256(store.image_path(clip))
        if progress is not None:
            progress("rgb_hashes", i + 1, len(store.clips))
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "PREPARATION_INCOMPLETE"
    incomplete.write_text("All-frame manifests are unavailable until experiments.json is published.\n")
    manifests = build_all_frame_manifests(
        store, configs, shard_hashes=hashes,
        progress=(lambda done, total: progress("annotations", done, total)) if progress is not None else None,
    )
    datasets: dict[str, dict[str, Any]] = {}
    for length, modes in manifests.items():
        datasets[str(length)] = {}
        for mode, manifest in modes.items():
            validate_all_frame_manifest(manifest, store)
            path = output / f"{mode}_T{length}.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            datasets[str(length)][mode] = dict(path=str(path), sha256=dual_sha256(path), summaries=manifest["summaries"])
        if progress is not None:
            progress("manifests", list(manifests).index(length) + 1, len(manifests))
    for split in ("train", "val", "test"):
        reference = next(iter(manifests.values()))["evaluation"]["summaries"][split]
        for modes in manifests.values():
            for manifest in modes.values():
                summary = manifest["summaries"][split]
                for name in ("common_intervals_sha256", "target_identity_sha256", "teacher_identity_sha256",
                             "scored_frame_sha256", "training_packs_sha256"):
                    if summary[name] != reference[name]:
                        raise ValueError(f"All-frame identity differs between T/modes: {split}/{name}")
    if any(file_version(path) != version for path, version in versions.items()):
        raise ValueError("Source files changed during all-frame preparation")
    if source_identity()["source_sha256"] != code["source_sha256"]:
        raise ValueError("All-frame implementation changed during preparation")
    plan = dict(schema="ball_all_frame_preparation.v1", datasets=datasets, code=code,
                configs=[asdict(c) for c in configs], snapshot_mode="read_only_reference; no copy or hardlinks",
                payload_mode="external_source_view" if external_payloads is not None else "self_contained",
                rgb_manifest=None if rgb_manifest is None else dict(path=str(rgb_manifest), sha256=dual_sha256(rgb_manifest)),
                source_files_stat_checked=len(versions), status="prepared; no training started")
    (output / "experiments.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    incomplete.unlink()
    return plan
