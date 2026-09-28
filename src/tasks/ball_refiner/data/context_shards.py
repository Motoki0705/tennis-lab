"""Pinned whole-clip generation and exact, immutable context-cache assembly."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import shard_name
from src.tasks.ball_refiner.data.context_cache import (
    ContextCache,
    ContextProducer,
    context_cache_identity,
    generate_context_cache,
)
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.utils.checksum import dual_sha256

PLAN_SCHEMA = "ball_refiner_context_shards.v1"


def _shards(evidence: EvidenceCache) -> list[dict[str, Any]]:
    # Never split a tracking sequence, select by labels, or pack unrelated clips.
    return [{"index": index, "clip_id": record["clip"]["clip_id"],
             "source": record["clip"]["source"], "frames": record["clip"]["frame_count"],
             "jpeg_shard_sha256": record["jpeg_shard_sha256"]}
            for index, record in enumerate(evidence.manifest["clips"])]


def write_context_plan(evidence: EvidenceCache, *, producer: ContextProducer, output: Path) -> Path:
    """Pin all detector clips and model/code identity before any queued inference."""
    if not output.is_absolute():
        raise ValueError("Context plan output must be absolute")
    if output.exists():
        raise FileExistsError(f"Context plan output already exists: {output}")
    plan = {
        "schema": PLAN_SCHEMA,
        "identity": {**context_cache_identity(evidence), "model_identity": json_value(producer.identity())},
        "shards": _shards(evidence),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    write_json_atomic(path, plan)
    return path


def read_context_plan(path: Path, evidence: EvidenceCache) -> dict[str, Any]:
    """Validate complete ordered coverage and current input identity on every use."""
    if not path.is_absolute():
        raise ValueError("Context plan path must be absolute")
    digest = dual_sha256(path)
    plan: dict[str, Any] = json.loads(path.read_text())
    if (set(plan) != {"schema", "identity", "shards"} or plan["schema"] != PLAN_SCHEMA
            or plan["shards"] != _shards(evidence)):
        raise ValueError("Context plan must cover every detector clip exactly once in order")
    identity = plan["identity"]
    current = context_cache_identity(evidence)
    if (not isinstance(identity, dict) or set(identity) != {*current, "model_identity"}
            or not isinstance(identity["model_identity"], dict)
            or any(identity[key] != value for key, value in current.items())):
        raise ValueError("Context plan input/generator identity changed")
    if dual_sha256(path) != digest:
        raise ValueError("Context plan changed during read")
    return plan


def generate_context_shard(
    evidence: EvidenceCache, *, plan_path: Path, shard_index: int,
    producer: ContextProducer, output: Path,
) -> Path:
    plan = read_context_plan(plan_path, evidence)
    if type(shard_index) is not int or not 0 <= shard_index < len(plan["shards"]):
        raise ValueError("Shard index must address one whole clip in the plan")
    return generate_context_cache(
        evidence, output=output, producer=producer,
        clip_ids=(plan["shards"][shard_index]["clip_id"],), expected_identity=plan["identity"],
    )


def merge_context_shards(
    evidence: EvidenceCache, *, plan_path: Path, shards: tuple[Path, ...], output: Path,
) -> Path:
    """Copy verified NPZ bytes; publish a normal ContextCache only at full coverage.

    Input paths are explicit, including replacement attempts. No directory scan,
    latest-attempt preference, partial recovery, hard links or in-place mutation.
    Source caches remain usable independently and retain their failure history.
    """
    if not output.is_absolute():
        raise ValueError("Merged context output must be absolute")
    if output.exists():
        raise FileExistsError(f"Merged context output already exists: {output}")
    plan_digest = dual_sha256(plan_path)
    plan = read_context_plan(plan_path, evidence)
    if len(shards) != len(plan["shards"]) or len({path.resolve() for path in shards}) != len(shards):
        raise ValueError("Assembly requires one unique shard per planned clip")
    caches: dict[str, ContextCache] = {}
    receipts = []
    for path in shards:
        manifest_path = path / "manifest.json"
        digest = dual_sha256(manifest_path)
        cache = ContextCache(path, evidence)
        if any(cache.manifest.get(key) != value for key, value in plan["identity"].items()):
            raise ValueError(f"Shard model/code/input identity differs from the plan: {path}")
        if len(cache.clip_ids) != 1 or cache.clip_ids[0] in caches:
            raise ValueError("Each planned clip must be covered by exactly one whole-clip shard")
        if dual_sha256(manifest_path) != digest:
            raise ValueError("Shard manifest changed during assembly preflight")
        caches[cache.clip_ids[0]] = cache
        receipts.append({"directory": str(path), "manifest_sha256": digest, "clip_id": cache.clip_ids[0]})
    if set(caches) != set(evidence.clip_ids):
        raise ValueError("Shards do not cover the complete detector selection")
    manifest: dict[str, Any] = {
        **copy.deepcopy(plan["identity"]), "status": "building",
        "selection": {"clip_ids": list(evidence.clip_ids), "scope": "all_evidence"},
        "assembly": {"schema": PLAN_SCHEMA, "plan": {"path": str(plan_path), "sha256": plan_digest},
                     "shards": sorted(receipts, key=lambda item: evidence.clip_ids.index(item["clip_id"]))},
        "clips": [],
    }
    (output / "clips").mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "manifest.json", manifest)
    for clip_id in evidence.clip_ids:
        cache = caches[clip_id]
        cache.load(clip_id)  # Shapes, checksum, complete execution receipt and exact frame/PTS.
        record = copy.deepcopy(cache.manifest["clips"][0])
        jpeg = evidence.store.directory / "shards" / shard_name(record["clip"]["index"])
        if dual_sha256(jpeg) != record["jpeg_shard_sha256"]:
            raise ValueError(f"JPEG shard changed before assembly: {clip_id}")
        source, destination = cache.directory / record["file"], output / record["file"]
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if dual_sha256(destination) != record["sha256"] or dual_sha256(source) != record["sha256"]:
            raise ValueError(f"Context NPZ changed during assembly: {clip_id}")
        manifest["clips"].append(record)
        write_json_atomic(output / "manifest.json", manifest)
    # Recheck after the final copy, including clips read early in the assembly.
    for record in manifest["clips"]:
        jpeg = evidence.store.directory / "shards" / shard_name(record["clip"]["index"])
        if (dual_sha256(jpeg) != record["jpeg_shard_sha256"]
                or dual_sha256(output / record["file"]) != record["sha256"]):
            raise ValueError("JPEG or assembled NPZ changed before publication")
    for receipt in receipts:
        if dual_sha256(Path(receipt["directory"]) / "manifest.json") != receipt["manifest_sha256"]:
            raise ValueError("Shard manifest changed before publication")
    if dual_sha256(plan_path) != plan_digest or read_context_plan(plan_path, evidence) != plan:
        raise ValueError("Context plan changed before publication")
    manifest["status"] = "complete"
    write_json_atomic(output / "manifest.json", manifest)
    return output
