"""Explicit negative supervision and frozen windows outside the play selector."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import (
    AnnotationStates,
    annotation_states,
)
from src.tasks.ball_detection.data.store import (
    POINT_KIND_CODES,
    BallFrameStore,
    ClipRecord,
    shard_name,
)
from src.tasks.ball_detection.data.windows.intervals import (
    PlayIntervalConfig,
    mask_intervals,
)
from src.tasks.ball_detection.data.windows.manifest import (
    COORDINATE_MANIFEST_SCHEMA,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import (
    SampledWindow,
    TemporalSamplingConfig,
    select_sampled_windows,
)
from src.utils.checksum import dual_sha256

HYBRID_MANIFEST_SCHEMA = "ball_hybrid_windows.v1"
WINDOW_SOURCES = ("play", "absent", "nonplay")


@dataclass(frozen=True)
class AbsencePolicy:
    negative_kinds: tuple[str, ...]
    window_weights: dict[str, float]

    def __post_init__(self) -> None:
        if (len(set(self.negative_kinds)) != len(self.negative_kinds)
                or not set(self.negative_kinds) <= {"reviewed_empty", "out_of_frame"}):
            raise ValueError("Negative kinds must explicitly select reviewed_empty and/or out_of_frame")
        if set(self.window_weights) != set(WINDOW_SOURCES):
            raise ValueError("Declare all play/absent/nonplay window weights, including zeros")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in self.window_weights.values()):
            raise ValueError("Window weights must be finite and nonnegative")
        if not math.isclose(sum(self.window_weights.values()), 1., abs_tol=1e-9):
            raise ValueError("Window weights must sum to one; no automatic normalization")
        if not self.negative_kinds and self.window_weights["absent"]:
            raise ValueError("Absent windows require an explicitly selected negative kind")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AbsencePolicy:
        if set(raw) != {"negative_kinds", "window_weights"}:
            raise ValueError("Absence policy requires negative_kinds and window_weights exactly")
        return cls(tuple(raw["negative_kinds"]), dict(raw["window_weights"]))


def reviewed_empty_evidence(store: BallFrameStore, clip: ClipRecord, states: AnnotationStates) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Verify Chat notes against the hashed original; freeze the mask, not guessed labels."""
    empty = states.reviewed & states.target & (states.count == 0)
    if clip.source != "chat_annotation" or not empty.any():
        return empty, dict(rule="normalized reviewed target with zero instances", source=clip.source)
    try:
        path = store.annotation_path(clip)
    except ValueError as error:
        raise ValueError("Chat reviewed-empty annotation missing or checksum differs from the snapshot") from error
    if dual_sha256(path) != clip.annotation_sha256:
        raise ValueError("Chat reviewed-empty annotation missing or checksum differs from the snapshot")
    raw = json.loads(path.read_text())
    if (raw["schema_version"] != "tennis_chat_ball_annotation.v1" or f"chat_annotation/{raw['clip_id']}" != clip.clip_id
            or raw["frame_count"] != clip.frame_count or len(raw["frames"]) != clip.frame_count):
        raise ValueError("Chat reviewed-empty annotation identity differs from the snapshot")
    rows = store.clip_rows(clip)
    for i, frame in enumerate(raw["frames"]):
        if (frame["frame_index"] != int(store.frames["frame_index"][rows[i]])
                or frame["reviewed"] != bool(states.reviewed[i]) or len(frame["balls"]) != states.count[i]
                or not isinstance(frame["notes"], str)):
            raise ValueError("Chat reviewed-empty frame identity differs from the snapshot")
        empty[i] &= not bool(frame["notes"].strip())
    return empty, dict(rule="reviewed target, zero instances, no frame notes; verified original annotation",
                       annotation_sha256=clip.annotation_sha256, annotation_path=str(path),
                       excluded_noted_empty_frames=int((states.reviewed & states.target & (states.count == 0)).sum() - empty.sum()))


def negative_mask(states: AnnotationStates, policy: AbsencePolicy, reviewed_empty: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Unlocated, estimated, unreviewed and reference frames are never negatives."""
    negative = np.zeros_like(states.reviewed)
    if "reviewed_empty" in policy.negative_kinds:
        if reviewed_empty.shape != states.count.shape or (reviewed_empty & (states.count != 0)).any():
            raise ValueError("Reviewed-empty evidence must refer only to empty frames")
        negative |= reviewed_empty
    if "out_of_frame" in policy.negative_kinds:
        negative |= (states.count == 1) & (states.kinds[:, POINT_KIND_CODES["out_of_frame"]] == 1)
    return negative & states.reviewed & states.target


def eligible_spans(states: AnnotationStates, config: PlayIntervalConfig) -> list[tuple[int, int]]:
    """Keep reference and native-PTS barriers even when inputs are subsampled."""
    differences = np.diff(states.times)
    cuts = np.flatnonzero(differences > config.timestamp_gap_factor * np.median(differences)) + 1 if len(differences) else []
    spans: list[tuple[int, int]] = []
    for start, stop in mask_intervals(states.target):
        bounds = [start, *(int(c) for c in cuts if start < c < stop), stop]
        spans.extend(zip(bounds[:-1], bounds[1:], strict=True))
    return spans


def build_hybrid_manifest(base_path: Path, policy: AbsencePolicy) -> dict[str, Any]:
    """Extend an existing frozen snapshot, without copying or changing any inputs.

    Validation enumerates every accepted bucket. Training alone applies the
    explicit window mixture. Nonplay is a sampling region, never an absence label.
    """
    base: dict[str, Any] = json.loads(base_path.read_text())
    if base["schema"] != COORDINATE_MANIFEST_SCHEMA or base["input_kind"] != "mdd_only":
        raise ValueError("Hybrid preparation requires a frozen MDD-only coordinate manifest")
    root = Path(base["ball_store"]["directory"])
    for name, digest in base["ball_store"]["hashes"].items():
        if dual_sha256(root / name) != digest:
            raise ValueError("Ball snapshot changed")
    store = BallFrameStore(root)
    config = PlayIntervalConfig(**base["config"])
    sampling = TemporalSamplingConfig(tuple(base["sampling"]["frame_steps"]))
    known = {r["clip"]["clip_id"]: r for r in base["clips"]}
    common = base["common_evaluation"]
    common_ids = set(common["clip_ids"]) if common is not None else set()
    records = []
    counts: dict[str, dict[str, int]] = {}
    for clip in store.clips:
        states = annotation_states(store, clip)
        play, original = select_sampled_windows(states, config, sampling)
        previous = known.get(clip.clip_id)
        if previous is not None and previous["windows"] != [asdict(w) for w in original]:
            raise ValueError("Base manifest play windows disagree with the frozen annotations")
        play_mask: NDArray[np.bool_] = np.zeros(clip.frame_count, dtype=np.bool_)
        for start, stop in play:
            play_mask[start:stop] = True
        empty, evidence = (reviewed_empty_evidence(store, clip, states) if "reviewed_empty" in policy.negative_kinds
                           else (np.zeros_like(states.reviewed), dict(rule="reviewed_empty disabled")))
        negative = negative_mask(states, policy, empty)
        supervised = states.supervision | negative
        windows = list(original)
        sources = ["play"] * len(windows)
        seen = {(w.start, w.frame_step) for w in windows}
        spans = eligible_spans(states, config)
        for step in sampling.frame_steps:
            for start, stop in spans:
                last = stop - (31 * step + 1)
                if last < start:
                    continue
                candidates = sorted(set([*range(start, last + 1, config.window_stride * step), last]))
                for first in candidates:
                    window = SampledWindow(first, step)
                    if (first, step) in seen:
                        continue
                    indices = window.indices()
                    if negative[indices].all():
                        source = "absent"
                    elif not play_mask[first:indices[-1] + 1].any() and supervised[indices].any():
                        source = "nonplay"
                    else:
                        continue
                    windows.append(window)
                    sources.append(source)
                    seen.add((first, step))
        if not windows:
            continue
        semantic = clip_semantic_digest(store, clip)
        if previous is not None and semantic != previous["semantic_sha256"]:
            raise ValueError("Base snapshot clip identity changed")
        if clip.clip_id in common_ids and semantic != common["clip_semantic_sha256"][clip.clip_id]:
            raise ValueError("Common evaluation identity changed")
        digest = previous["rgb_shard_sha256"] if previous else dual_sha256(root / "shards" / shard_name(clip.index))
        records.append(dict(clip=asdict(clip), semantic_sha256=semantic, rgb_shard_sha256=digest,
                            play=play, eligible=spans, windows=[asdict(w) for w in windows],
                            reviewed_empty_frames=np.flatnonzero(empty).tolist(), negative_evidence=evidence,
                            window_sources=sources, common_evaluation=clip.clip_id in common_ids))
        for window, source in zip(windows, sources, strict=True):
            group = counts.setdefault(f"{clip.source}/{clip.split}/{window.frame_step}", dict.fromkeys(WINDOW_SOURCES, 0))
            group[source] += 1
    result = copy.deepcopy(base)
    result.update(schema=HYBRID_MANIFEST_SCHEMA, clips=records, counts=counts, absence=asdict(policy),
                  base_manifest_sha256=dual_sha256(base_path))
    return result
