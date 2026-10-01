"""Pinned Meiji-only plans and whole-clip resume without reusing partial inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord, shard_name
from src.tasks.ball_refiner.data.context_arrays import GeneratedContext
from src.tasks.ball_refiner.data.context_cache import (
    ContextCache,
    ContextProducer,
    context_cache_identity,
    generate_context_cache,
)
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.utils.checksum import dual_sha256

SCHEMA = "ball_refiner_meiji_context_plan.v1"


def selection(evidence: EvidenceCache) -> tuple[list[str], list[dict[str, Any]]]:
    selected, absent = [], []
    for name in evidence.clip_ids:
        clip = evidence.store.clip_by_id(name)
        if clip.split not in {"train", "val"}:
            raise ValueError("Context plan cannot reference test clips")
        if clip.source == "meiji":
            expected = "video_002" if clip.split == "train" else "video_000"
            if f"/{expected}/" not in name or clip.camera_id is None:
                raise ValueError("Meiji video/split/camera identity mismatch")
            selected.append(name)
        else:
            absent.append({"clip_id": name, "source": clip.source, "status": "absent_by_policy",
                           "pose": "absent", "court": "absent", "reason": "meiji_only_user_decision"})
    if not selected:
        raise ValueError("No Meiji train/val clips")
    return selected, absent


def write_meiji_plan(evidence: EvidenceCache, producer: ContextProducer, output: Path) -> Path:
    if not output.is_absolute() or output.exists():
        raise ValueError("Plan needs a new absolute output directory")
    selected, absent = selection(evidence)
    records = {r["clip"]["clip_id"]: r for r in evidence.manifest["clips"]}
    plan = {"schema": SCHEMA, "identity": {**context_cache_identity(evidence), "model_identity": json_value(producer.identity())},
            "clips": [{"clip_id": name, "frames": evidence.store.clip_by_id(name).frame_count,
                       "jpeg_shard_sha256": records[name]["jpeg_shard_sha256"]} for name in selected],
            "absent": absent,
            "budget": {"wall_seconds": 43170, "vram_stop_bytes": 9_500_000_000,
                       "allocator_bytes": 7 * 1024**3, "disk_limit_bytes": 10_000_000_000},
            "joint_contract": "COCO17_all_raw_finite_peaks; missing_mask; no_GSI_observations"}
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "plan.json", plan)
    return output / "plan.json"


def validate_plan(path: Path, digest: str, evidence: EvidenceCache, producer: ContextProducer) -> dict[str, Any]:
    if dual_sha256(path) != digest:
        raise ValueError("Context plan hash mismatch")
    plan: dict[str, Any] = json.loads(path.read_text())
    selected, absent = selection(evidence)
    identity = {**context_cache_identity(evidence), "model_identity": json_value(producer.identity())}
    records = {r["clip"]["clip_id"]: r for r in evidence.manifest["clips"]}
    expected = [{"clip_id": name, "frames": evidence.store.clip_by_id(name).frame_count,
                 "jpeg_shard_sha256": records[name]["jpeg_shard_sha256"]} for name in selected]
    if plan["schema"] != SCHEMA or plan["identity"] != identity or plan["clips"] != expected or plan["absent"] != absent:
        raise ValueError("Context plan input/model/code/selection identity mismatch")
    for row in plan["clips"]:
        clip = evidence.store.clip_by_id(row["clip_id"])
        if dual_sha256(evidence.store.directory / "shards" / shard_name(clip.index)) != row["jpeg_shard_sha256"]:
            raise ValueError(f"Context JPEG input changed: {clip.clip_id}")
    return plan


class CompletedClipProducer:
    """Explicit assembly from verified receipts; never calls an inference model."""
    def __init__(self, identity: dict[str, Any], caches: dict[str, ContextCache]) -> None:
        self.pinned, self.caches = identity, caches

    def identity(self) -> dict[str, Any]:
        return self.pinned

    def predict(self, store: BallFrameStore, clip: ClipRecord) -> GeneratedContext:
        result: GeneratedContext = self.caches[clip.clip_id].load(clip.clip_id)
        return result


def resume_meiji_cache(evidence: EvidenceCache, producer: ContextProducer, plan_path: Path, plan_sha256: str) -> Path:
    """Resume complete clips only; failed/interrupted attempts are retained and labelled.

    Runtime failures stop this job. A later explicitly launched resume retries
    the whole incomplete clip with identical inputs in a new attempt directory.
    All other sources remain explicitly absent in the coverage manifest.
    """
    plan = validate_plan(plan_path, plan_sha256, evidence, producer)
    root = plan_path.parent
    progress_path = root / "progress.json"
    progress: dict[str, Any] = (json.loads(progress_path.read_text()) if progress_path.exists() else
                {"plan_sha256": plan_sha256, "status": "running", "attempts": [], "completed": {}})
    if progress["plan_sha256"] != plan_sha256:
        raise ValueError("Resume receipt belongs to another plan")
    caches: dict[str, ContextCache] = {}
    selected = [r["clip_id"] for r in plan["clips"]]
    if not set(progress["completed"]) <= set(selected):
        raise ValueError("Resume contains an unplanned clip")
    for name, receipt in progress["completed"].items():
        directory = root / receipt["directory"]
        if not directory.resolve().is_relative_to(root.resolve()) or dual_sha256(directory / "manifest.json") != receipt["sha256"]:
            raise ValueError("Completed clip receipt/hash mismatch")
        cache = ContextCache(directory, evidence)
        if cache.clip_ids != (name,) or any(cache.manifest[k] != v for k, v in plan["identity"].items()):
            raise ValueError("Completed clip identity mismatch")
        cache.load(name)
        caches[name] = cache
    for attempt in progress["attempts"]:
        if attempt["status"] == "running":
            attempt.update(status="interrupted", frame_status="not_published", frame_range=[0, attempt["frames"]])
    progress["status"] = "running"
    write_json_atomic(progress_path, progress)
    for row in plan["clips"]:
        name = row["clip_id"]
        if name in caches:
            continue
        attempt = {"clip_id": name, "frames": row["frames"], "status": "running",
                   "directory": f"attempts/{len(progress['attempts']):05d}",
                   "frame_status": "not_published", "frame_range": [0, row["frames"]]}
        progress["attempts"].append(attempt)
        write_json_atomic(progress_path, progress)
        directory = root / attempt["directory"]
        try:
            generate_context_cache(evidence, output=directory, producer=producer, clip_ids=(name,), expected_identity=plan["identity"])
            cache = ContextCache(directory, evidence)
            cache.load(name)
        except BaseException as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}",
                           last_stage=json_value(getattr(producer, "progress", {})), frame_status="failed_not_published")
            progress["status"] = "failed"
            write_json_atomic(progress_path, progress)
            raise
        attempt.update(status="complete", frame_status="see_pose_frame_status_in_clip_receipt")
        progress["completed"][name] = {"directory": attempt["directory"], "sha256": dual_sha256(directory / "manifest.json")}
        caches[name] = cache
        write_json_atomic(progress_path, progress)
    validate_plan(plan_path, plan_sha256, evidence, producer)
    # A killed assembly can be retried without touching or inferring any clip.
    assemblies = progress.setdefault("assemblies", [])
    output: Path
    if progress.get("cache") is not None:
        output = root / progress["cache"]
        cache = ContextCache(output, evidence)
        if dual_sha256(output / "manifest.json") != progress["cache_sha256"]:
            raise ValueError("Published cache manifest changed")
        for name in selected:
            cache.load(name)
    else:
        output = root / f"assembly-{len(assemblies):03d}"
        assemblies.append(str(output.relative_to(root)))
        write_json_atomic(progress_path, progress)
        generate_context_cache(evidence, output=output,
            producer=CompletedClipProducer(plan["identity"]["model_identity"], caches),
            clip_ids=tuple(selected), expected_identity=plan["identity"])
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["coverage"] = {"policy": "meiji_only", "absent": plan["absent"], "plan_sha256": plan_sha256}
        manifest["assembly_receipts"] = progress["completed"]
        write_json_atomic(output / "manifest.json", manifest)
        progress.update(cache=str(output.relative_to(root)), cache_sha256=dual_sha256(output / "manifest.json"))
    validate_plan(plan_path, plan_sha256, evidence, producer)
    progress["status"] = "complete"
    write_json_atomic(progress_path, progress)
    return output
