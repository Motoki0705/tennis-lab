"""Count all supervision and context states without selecting an easier subset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context import load_pipeline_context, meiji_scene_index
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.utils.checksum import dual_sha256


def audit_store(store: BallFrameStore, *, meiji_context_root: Path, pose_threshold: float) -> dict[str, Any]:
    """Inspect labels and every Meiji camera; corrupt context raises, never masks.

    Absence is reported for source clips with no generated context. Original
    videos/JPEG shards are not rehashed or inferred by this CPU-only audit.
    """
    if not meiji_context_root.is_dir():
        raise FileNotFoundError(meiji_context_root)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_counts: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], str] = {}
    for clip in store.clips:
        group = (clip.source, clip.group_id)
        if group in groups and groups[group] != clip.split:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = clip.split
        targets = project_store_targets(store, clip)
        key = f"{clip.source}/{clip.split}"
        counts[key].update(targets.counts())
        counts[key].update(clips=1, frames=clip.frame_count,
                           position_known=int(targets.position_valid.sum()),
                           presence_known=int(targets.presence_valid.sum()))
        record: dict[str, Any] = {"clip": asdict(clip), "targets": targets.counts()}
        if clip.source == "meiji":
            context = load_pipeline_context(
                clip, meiji_scene_index(clip, meiji_context_root), pose_threshold=pose_threshold,
            )
            record["context"] = context.provenance
            for component in ("pose", "court"):
                state = str(context.provenance[f"{component}_status"])
                context_counts[key][f"{component}_{state}"] += 1
            context_counts[key]["complete"] += int(context.pose is not None and context.court is not None)
        else:
            record["context"] = {"pose_status": "not_audited", "court_status": "not_audited"}
        records.append(record)
    return {
        "schema": "ball_refiner_data_audit.v1",
        "store": str(store.directory.resolve()),
        "store_sha256": {name: dual_sha256(store.directory / name) for name in ("metadata.json", "index.npz")},
        "context_root": str(meiji_context_root.resolve()),
        "scope": "store labels and indexed Meiji context; no RGB decode, inference or original-media rehash",
        "counts": {key: dict(value) for key, value in sorted(counts.items())},
        "context_counts": {key: dict(value) for key, value in sorted(context_counts.items())},
        "clips": records,
    }
