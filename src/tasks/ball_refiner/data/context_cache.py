"""Immutable context caches bound to the detector's exact decoded JPEG shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord, shard_name
from src.tasks.ball_refiner.data.cache_identity import clip_record, store_hashes
from src.tasks.ball_refiner.data.context_arrays import ContextArrays, GeneratedContext
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

SCHEMA = "ball_refiner_context.v1"
RGB_CONDITION = "unmodified_stored_jpeg_bgr.v1"
COORDINATES = "stored_jpeg_pixels; source_xy=stored_xy/clip.scale"


class ContextProducer(Protocol):
    def identity(self) -> dict[str, Any]:
        """Rehash assets/code and return the complete inference configuration."""
        ...

    def predict(self, store: BallFrameStore, clip: ClipRecord) -> GeneratedContext:
        """Execute all frames from the store; do not catch model failures."""
        ...


def _assert_timeline(result: ContextArrays, store: BallFrameStore, clip: ClipRecord) -> None:
    rows = store.clip_rows(clip)
    if not np.array_equal(result.frame_index, store.frames["frame_index"][rows]) or not np.array_equal(
        result.pts, store.frames["pts"][rows],
    ):
        raise ValueError(f"Context frame/PTS mismatch: {clip.clip_id}")
    points = result.court_points[result.court_valid]
    if (points < 0).any() or (points > np.asarray([clip.width - 1, clip.height - 1])).any():
        raise ValueError(f"Valid court points must be inside the stored image: {clip.clip_id}")


def generate_context_cache(
    evidence: EvidenceCache, *, output: Path, producer: ContextProducer,
    clip_ids: tuple[str, ...] | None,
) -> Path:
    """Publish every completed clip, then mark complete after final hash checks.

    ``None`` explicitly means every clip in the detector cache. A pilot can
    name a subset, retained in the manifest and never advertised as full
    coverage. This schema accepts only unmodified JPEG input; RGB occlusion
    needs a separately identified cache of both detector and context evidence.
    """
    if not output.is_absolute():
        raise ValueError("Context output must be absolute")
    if output.exists():
        raise FileExistsError(f"Context output already exists: {output}")
    selected = evidence.clip_ids if clip_ids is None else clip_ids
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(evidence.clip_ids):
        raise ValueError("Context selection must name unique clips from the detector cache")
    selected = tuple(clip_id for clip_id in evidence.clip_ids if clip_id in set(selected))
    store = evidence.store
    store_identity = store_hashes(store.directory)
    if store_identity != evidence.manifest["store"]["sha256"]:
        raise ValueError("Store changed after opening detector evidence")
    evidence_path = evidence.directory / "manifest.json"
    evidence_hash = dual_sha256(evidence_path)
    if json.loads(evidence_path.read_text()) != evidence.manifest:
        raise ValueError("Detector manifest changed after opening evidence")
    detector_records = {record["clip"]["clip_id"]: record for record in evidence.manifest["clips"]}
    identity = producer.identity()
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "building", "rgb_condition": RGB_CONDITION,
        "coordinate_system": COORDINATES, "model_identity": identity,
        "generator_sha256": {name: dual_sha256(Path(__file__).with_name(name)) for name in (
            "context_cache.py", "context_arrays.py", "cache_identity.py",
        )},
        "store": {"directory": str(store.directory), "sha256": store_identity},
        "evidence": {"directory": str(evidence.directory), "manifest_sha256": evidence_hash},
        "selection": {"clip_ids": list(selected), "scope": "all_evidence" if clip_ids is None else "explicit_subset"},
        "clips": [],
    }
    (output / "clips").mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "manifest.json", manifest)
    for clip_id in selected:
        clip = store.clip_by_id(clip_id)
        shard = store.directory / "shards" / shard_name(clip.index)
        shard_hash = detector_records[clip_id]["jpeg_shard_sha256"]
        if dual_sha256(shard) != shard_hash:
            raise ValueError(f"Context RGB differs from detector JPEG input: {clip_id}")
        result = producer.predict(store, clip)
        _assert_timeline(result.arrays, store, clip)
        if dual_sha256(shard) != shard_hash:
            raise ValueError(f"JPEG shard changed during context inference: {clip_id}")
        relative = f"clips/clip-{clip.index:05d}.npz"
        with (output / relative).open("xb") as stream:
            np.savez_compressed(stream, **result.arrays.arrays())
        manifest["clips"].append({
            "clip": clip_record(clip), "file": relative, "sha256": dual_sha256(output / relative),
            "jpeg_shard_sha256": shard_hash, "execution": result.execution,
        })
        write_json_atomic(output / "manifest.json", manifest)
        print(json.dumps({"clip_id": clip_id, "completed": len(manifest["clips"]), "total": len(selected),
                          "frames": clip.frame_count, "tracks": len(result.arrays.track_ids)}), flush=True)
    for record in manifest["clips"]:
        shard = store.directory / "shards" / shard_name(record["clip"]["index"])
        if dual_sha256(shard) != record["jpeg_shard_sha256"]:
            raise ValueError("A completed clip's JPEG shard changed before publication")
    if (store_hashes(store.directory) != store_identity or dual_sha256(evidence_path) != evidence_hash
            or producer.identity() != identity):
        raise ValueError("Context store, detector manifest, model assets or code changed during generation")
    manifest["status"] = "complete"
    write_json_atomic(output / "manifest.json", manifest)
    return output


class ContextCache:
    def __init__(self, directory: Path, evidence: EvidenceCache) -> None:
        if not directory.is_absolute():
            raise ValueError("Context cache directory must be absolute")
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["schema"] != SCHEMA or manifest["status"] != "complete"
                or manifest["rgb_condition"] != RGB_CONDITION or manifest["coordinate_system"] != COORDINATES):
            raise ValueError("Incomplete or incompatible context cache")
        if (manifest["store"]["sha256"] != store_hashes(evidence.store.directory)
                or manifest["evidence"]["manifest_sha256"] != dual_sha256(evidence.directory / "manifest.json")):
            raise ValueError("Context does not belong to this store/detector evidence")
        selected = manifest["selection"]["clip_ids"]
        scope = manifest["selection"]["scope"]
        if (not selected or scope not in {"all_evidence", "explicit_subset"}
                or selected != [item for item in evidence.clip_ids if item in set(selected)]
                or (scope == "all_evidence" and selected != list(evidence.clip_ids))):
            raise ValueError("Context selection is not an exact ordered detector-cache subset")
        records = manifest["clips"]
        if [record["clip"]["clip_id"] for record in records] != selected:
            raise ValueError("Context cache does not cover every declared clip exactly once")
        detector_records = {record["clip"]["clip_id"]: record for record in evidence.manifest["clips"]}
        for record in records:
            clip = evidence.store.clip_by_id(record["clip"]["clip_id"])
            if (record["clip"] != clip_record(clip) or record["file"] != f"clips/clip-{clip.index:05d}.npz"
                    or record["jpeg_shard_sha256"] != detector_records[clip.clip_id]["jpeg_shard_sha256"]):
                raise ValueError(f"Context metadata/RGB/path mismatch: {clip.clip_id}")
        self.directory, self.evidence = directory, evidence
        self.clip_ids = tuple(selected)
        self.manifest: dict[str, Any] = manifest
        self._records = {record["clip"]["clip_id"]: record for record in records}

    def require_clips(self, clip_ids: tuple[str, ...]) -> None:
        """Training must call this before constructing any full/context dataset."""
        if not set(clip_ids) <= set(self.clip_ids):
            raise ValueError(f"Context was not generated for required clips: {sorted(set(clip_ids) - set(self.clip_ids))}")

    def load(self, clip_id: str) -> GeneratedContext:
        record = self._records[clip_id]
        path = self.directory / record["file"]
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("Context NPZ escapes the cache directory")
        digest = dual_sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Context checksum mismatch: {clip_id}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = ContextArrays.from_arrays({name: archive[name] for name in archive.files})
        if dual_sha256(path) != digest:
            raise ValueError(f"Context changed during read: {clip_id}")
        _assert_timeline(arrays, self.evidence.store, self.evidence.store.clip_by_id(clip_id))
        return GeneratedContext(arrays, record["execution"])
