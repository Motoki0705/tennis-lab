"""Versioned all-frame manifests, regenerated from a snapshot and its annotations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from fractions import Fraction
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frames import (
    EDGE_BINS,
    LABEL_KINDS,
    AllFrameWindowConfig,
    common_intervals,
    decode_label_runs,
    encode_label_runs,
    evaluation_windows,
    owner_starts,
    stored_label_kinds,
    training_packs,
    verified_label_kinds,
)
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.utils.checksum import dual_sha256

ALL_FRAME_MANIFEST_SCHEMA = "ball_all_frame_windows.v2"
ALL_FRAME_SEMANTICS = dict(
    mdd="recompute_from_each_augmented_child_rgb_window", timestamps="real_seconds",
    context="all_maximal_native_target_spans_at_least_max_window_length",
    training="uniform_shared_packs_with_replacement; retain_every_child",
    acceptance="train:at_least_one_observed_per_pack; validation/test:unchanged; no_child_filter",
    evaluation_stride="window_length//2", tail="append_last_start_once",
    owner="min(abs(2*frame-(2*start+T-1)),start)", score="owner_and_known_teacher",
    edge_distance="min(frame-interval_start,interval_stop-1-frame)", edge_bins=list(EDGE_BINS),
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def manifest_summaries(records: list[dict[str, Any]], config: AllFrameWindowConfig, mode: str) -> dict[str, Any]:
    """Frame keys use exact FPS; the full common and known-teacher sets are distinct."""
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        names = ("common_intervals", "target_identity", "teacher_identity", "scored_frame", "training_packs")
        digests = {name: hashlib.sha256() for name in names}
        owner_digest = hashlib.sha256()
        counts = dict.fromkeys(LABEL_KINDS, 0)
        interval_count = pack_count = window_count = clip_count = 0
        for record in records:
            clip = record["clip"]
            if clip["split"] != split:
                continue
            clip_count += 1
            clip_id = clip["clip_id"]
            intervals = [tuple(span) for span in record["common_intervals"]]
            codes = decode_label_runs(record["label_runs"], clip["frame_count"])
            digests["common_intervals"].update((_json([clip_id, intervals]) + "\n").encode())
            interval_count += len(intervals)
            for a, b in intervals:
                for f in range(a, b):
                    key = [clip_id, f, 1, str(Fraction(clip["fps"]))]
                    line = (_json(key) + "\n").encode()
                    digests["target_identity"].update(line)
                    digests["teacher_identity"].update((_json([*key, int(codes[f])]) + "\n").encode())
                    if codes[f]:
                        digests["scored_frame"].update(line)
                for name, count in zip(LABEL_KINDS, np.bincount(codes[a:b], minlength=4), strict=True):
                    counts[name] += int(count)
                if mode == "evaluation":
                    owner_digest.update((_json([clip_id, a, b, owner_starts(a, b, config.window_length).tolist()]) + "\n").encode())
            packs = training_packs(intervals, codes, config, require_position=split == "train")
            pack_count += len(packs)
            for pack in packs:
                digests["training_packs"].update((_json([clip_id, pack["start"], config.pack_length]) + "\n").encode())
            window_count += len(record["windows"]) if mode == "evaluation" else len(packs) * (config.pack_length // config.window_length)
        result[split] = dict(
            **{f"{name}_sha256": digest.hexdigest() for name, digest in digests.items()},
            counts=dict(clips=clip_count, intervals=interval_count, frames=sum(counts.values()),
                        scored_frames=sum(counts.values()) - counts["masked"], packs=pack_count,
                        windows=window_count, **counts),
        )
        if mode == "evaluation":
            result[split]["owner_identity_sha256"] = owner_digest.hexdigest()
    return result


def build_all_frame_manifests(
    store: BallFrameStore, configs: Sequence[AllFrameWindowConfig], *,
    shard_hashes: dict[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Select common intervals once; emit pack and owner manifests for each T.

    This entry reads normalized annotations and verifies original Chat notes.
    A caller may supply shard hashes from a verified manifest of this snapshot.
    No knowledge bundle, old play selection, image copying or links are used.
    """
    if not configs or len({c.window_length for c in configs}) != len(configs):
        raise ValueError("Declare nonempty, unique all-frame window lengths")
    selection = asdict(configs[0])
    selection.pop("window_length")
    if any({k: v for k, v in asdict(c).items() if k != "window_length"} != selection for c in configs):
        raise ValueError("All T must share common-interval and pack parameters")
    identity = {name: dual_sha256(store.directory / name) for name in ("metadata.json", "index.npz")}
    records, skipped = [], []
    group_splits: dict[tuple[str, str], str] = {}
    clips = sorted(store.clips, key=lambda c: c.clip_id)
    for i, clip in enumerate(clips):
        group = (clip.source, clip.group_id)
        if group_splits.setdefault(group, clip.split) != clip.split:
            raise ValueError("A source group crosses split boundaries")
        states = annotation_states(store, clip)
        spans = common_intervals(states, configs[0])
        if not spans:
            skipped.append(dict(clip_id=clip.clip_id, reason="no_common_interval"))
        else:
            codes, evidence = verified_label_kinds(store, clip, states)
            digest = shard_hashes[clip.clip_id] if shard_hashes is not None else dual_sha256(store.image_path(clip))
            records.append(dict(clip=asdict(clip), semantic_sha256=clip_semantic_digest(store, clip),
                                rgb_shard_sha256=digest, common_intervals=spans,
                                label_runs=encode_label_runs(codes), negative_evidence=evidence))
        if progress is not None:
            progress(i + 1, len(clips))
    if not records:
        raise ValueError("No common all-frame intervals in the snapshot")
    if any(dual_sha256(store.directory / name) != expected for name, expected in identity.items()):
        raise ValueError("Ball snapshot changed during all-frame preparation")
    manifests: dict[int, dict[str, dict[str, Any]]] = {}
    for config in configs:
        modes = {}
        for mode in ("train", "evaluation"):
            selected = copy.deepcopy(records)
            for record in selected:
                spans = record["common_intervals"]
                if mode == "train":
                    codes = decode_label_runs(record["label_runs"], record["clip"]["frame_count"])
                    record["packs"] = training_packs(spans, codes, config, require_position=record["clip"]["split"] == "train")
                else:
                    record["windows"] = evaluation_windows(spans, config.window_length)
            modes[mode] = dict(schema=ALL_FRAME_MANIFEST_SCHEMA, mode=mode, input_kind="mdd_only",
                               ball_store=dict(directory=str(store.directory), hashes=identity),
                               pose_directory=None, config=asdict(config), semantics=copy.deepcopy(ALL_FRAME_SEMANTICS),
                               label_kinds=list(LABEL_KINDS), clips=selected, skipped=skipped,
                               summaries=manifest_summaries(selected, config, mode))
        manifests[config.window_length] = modes
    return manifests


def validate_all_frame_manifest(manifest: dict[str, Any], store: BallFrameStore) -> AllFrameWindowConfig:
    """Reject relabeled legacy manifests and altered membership/owners before RGB I/O."""
    fields = {"schema", "mode", "input_kind", "ball_store", "pose_directory", "config", "semantics",
              "label_kinds", "clips", "skipped", "summaries"}
    if set(manifest) != fields or manifest["schema"] != ALL_FRAME_MANIFEST_SCHEMA:
        raise ValueError("Invalid all-frame manifest schema fields; legacy/new schemas cannot be mixed")
    if (manifest["mode"] not in {"train", "evaluation"} or manifest["input_kind"] != "mdd_only"
            or manifest["pose_directory"] is not None or manifest["semantics"] != ALL_FRAME_SEMANTICS
            or manifest["label_kinds"] != list(LABEL_KINDS)):
        raise ValueError("Unsupported all-frame manifest semantics")
    if set(manifest["config"]) != set(AllFrameWindowConfig.__dataclass_fields__):
        raise ValueError("All-frame configuration must declare its parameters exactly")
    config = AllFrameWindowConfig(**manifest["config"])
    if set(manifest["ball_store"]["hashes"]) != {"metadata.json", "index.npz"}:
        raise ValueError("All-frame snapshot must pin both index and metadata")
    records = manifest["clips"]
    ids = [r["clip"]["clip_id"] for r in records]
    if ids != sorted(set(ids)):
        raise ValueError("All-frame clips must be unique and sorted by clip_id")
    expected_ids, expected_skipped = [], []
    groups: dict[tuple[str, str], str] = {}
    for clip in sorted(store.clips, key=lambda c: c.clip_id):
        group = (clip.source, clip.group_id)
        if groups.setdefault(group, clip.split) != clip.split:
            raise ValueError("A source group crosses split boundaries")
        if common_intervals(annotation_states(store, clip), config):
            expected_ids.append(clip.clip_id)
        else:
            expected_skipped.append(dict(clip_id=clip.clip_id, reason="no_common_interval"))
    if ids != expected_ids or manifest["skipped"] != expected_skipped:
        raise ValueError("All-frame manifest must retain every common interval")
    for record in records:
        selection_key = "packs" if manifest["mode"] == "train" else "windows"
        if set(record) != {"clip", "semantic_sha256", "rgb_shard_sha256", "common_intervals",
                           "label_runs", "negative_evidence", selection_key}:
            raise ValueError("All-frame pack/evaluation records cannot contain legacy window fields")
        clip = store.clip_by_id(record["clip"]["clip_id"])
        if _json(asdict(clip)) != _json(record["clip"]) or record["semantic_sha256"] != clip_semantic_digest(store, clip):
            raise ValueError("Frozen all-frame clip identity changed")
        states = annotation_states(store, clip)
        spans = common_intervals(states, config)
        if _json(record["common_intervals"]) != _json(spans):
            raise ValueError("Frozen common intervals disagree with native target/PTS barriers")
        codes = decode_label_runs(record["label_runs"], clip.frame_count)
        empty = codes == 2
        if (empty & ~(states.reviewed & states.target & (states.count == 0))).any():
            raise ValueError("Frozen reviewed-empty labels contradict the snapshot")
        if not np.array_equal(codes, stored_label_kinds(store, clip, states, empty)):
            raise ValueError("Frozen teacher kinds disagree with normalized annotations")
        if (clip.source == "chat_annotation" and empty.any()
                and record["negative_evidence"].get("annotation_sha256") != clip.annotation_sha256):
            raise ValueError("Frozen Chat absence evidence lacks the verified annotation hash")
        expected = training_packs(spans, codes, config, require_position=clip.split == "train") if manifest["mode"] == "train" else evaluation_windows(spans, config.window_length)
        if record[selection_key] != expected:
            raise ValueError("Frozen pack membership or evaluation ownership changed")
    if manifest["summaries"] != manifest_summaries(records, config, manifest["mode"]):
        raise ValueError("All-frame membership hashes or counts disagree with the manifest")
    return config
