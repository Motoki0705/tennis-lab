"""CPU evidence for fixed central targets; never decodes or modifies source media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.data.windows.hybrid import (
    AbsencePolicy,
    eligible_spans,
    negative_mask,
    reviewed_empty_evidence,
)
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.tasks.ball_detection.data.windows.temporal import (
    TemporalSamplingConfig,
    select_sampled_windows,
)
from src.utils.checksum import dual_sha256

LENGTHS = (32, 64, 128, 256)
CENTRAL_LENGTH = 16
LABELS = ("masked", "observed", "reviewed_empty", "out_of_frame")
APIS = (
    "src/tasks/ball_detection/data/annotation_states.py",
    "src/tasks/ball_detection/data/store.py",
    "src/tasks/ball_detection/data/teacher_masks.py",
    "src/tasks/ball_detection/data/windows/hybrid.py",
    "src/tasks/ball_detection/data/windows/intervals.py",
    "src/tasks/ball_detection/data/windows/manifest.py",
    "src/tasks/ball_detection/data/windows/temporal.py",
)


def classify(store: BallFrameStore, clip: ClipRecord, states: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the existing hybrid negative policies, including hashed Chat notes."""
    empty, evidence = reviewed_empty_evidence(store, clip, states)
    kinds = {}
    for kind in LABELS[2:]:
        policy = AbsencePolicy((kind,), dict(play=1., absent=0., nonplay=0.))
        kinds[kind] = negative_mask(states, policy, empty)
    masks = (states.supervision, kinds["reviewed_empty"], kinds["out_of_frame"])
    if np.any(np.sum(masks, axis=0) > 1):
        raise ValueError("Teacher classes overlap")
    codes = np.zeros(clip.frame_count, np.uint8)
    for code, mask in enumerate(masks, 1):
        codes[mask] = code
    return codes, evidence


def context_starts(centers: list[int], length: int, spans: list[tuple[int, int]]) -> list[int]:
    margin = (length - CENTRAL_LENGTH) // 2
    if length < CENTRAL_LENGTH or (length - CENTRAL_LENGTH) % 2:
        raise ValueError("Context must extend the center symmetrically")
    return [c for c in centers if any(a <= c - margin and c + CENTRAL_LENGTH + margin <= b for a, b in spans)]


def compress_starts(starts: list[int]) -> list[list[int]]:
    """Inclusive arithmetic runs; irregular tail-backfill anchors are preserved."""
    if starts != sorted(set(starts)):
        raise ValueError("Central starts must be sorted and unique")
    runs: list[list[int]] = []
    for start in starts:
        if runs and start == runs[-1][1] + 16:
            runs[-1][1] = start
        else:
            runs.append([start, start, 16])
    if [s for a, b, step in runs for s in range(a, b + 1, step)] != starts:
        raise ValueError("Compressed centers changed membership")
    return runs


def usable_centers(codes: np.ndarray, starts: list[int], width: int) -> tuple[list[int], np.ndarray]:
    """A fully masked block cannot be sampled as a supervised training example."""
    accepted = [s for s in starts if np.any(codes[s:s + width])]
    mask: NDArray[np.bool_] = np.zeros(len(codes), np.bool_)
    for start in accepted:
        mask[start:start + width] = True
    return accepted, mask


def selection_stats(codes: np.ndarray, occluded: np.ndarray, starts: list[int], width: int) -> tuple[dict[str, int], np.ndarray]:
    mask: NDArray[np.bool_] = np.zeros(len(codes), np.bool_)
    result = dict(candidate_blocks=len(starts), supervised_blocks=0, negative_only_blocks=0, zero_teacher_blocks=0)
    occurrences: NDArray[np.int64] = np.zeros(4, np.int64)
    for start in starts:
        block = codes[start:start + width]
        if start < 0 or len(block) != width:
            raise ValueError("Block outside clip")
        mask[start:start + width] = True
        counts = np.bincount(block, minlength=4)
        occurrences += counts
        known = bool(np.any(block))
        result["supervised_blocks"] += int(known)
        result["zero_teacher_blocks"] += int(not known)
        result["negative_only_blocks"] += int(known and counts[1] == 0)
    for code, name in enumerate(LABELS):
        result[name] = int(np.sum(mask & (codes == code)))
        result[f"{name}_occurrences"] = int(occurrences[code])
    result["observed_occluded"] = int(np.sum(mask & (codes == 1) & occluded))
    result["unique_frames"] = int(mask.sum())
    if sum(result[k] for k in LABELS) != result["unique_frames"]:
        raise ValueError("Teacher partition changed")
    return result, mask


def fingerprint(paths: list[Path]) -> dict[str, tuple[int, int, int, int]]:
    return {str(p): (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for p in paths for s in [p.stat()]}


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def identity(clips: list[ClipRecord], masks: dict[int, np.ndarray], codes: dict[int, np.ndarray]) -> dict[str, str]:
    digests = {k: hashlib.sha256() for k in ("target_identity_sha256", "teacher_identity_sha256")}
    for clip in sorted(clips, key=lambda c: c.clip_id):
        for index in np.flatnonzero(masks[clip.index]):
            key = [clip.clip_id, int(index), 1, str(Fraction(clip.fps))]
            digests["target_identity_sha256"].update((json.dumps(key, separators=(",", ":")) + "\n").encode())
            digests["teacher_identity_sha256"].update((json.dumps([*key, int(codes[clip.index][index])], separators=(",", ":")) + "\n").encode())
    return {k: v.hexdigest() for k, v in digests.items()}


def survey(repo_root: Path, prior_path: Path, output: Path) -> None:
    began = time.perf_counter()
    if not output.resolve().is_relative_to(Path.cwd().resolve()) or output.exists():
        raise ValueError("Use a fresh output directory inside the active worktree")
    prior = json.loads(prior_path.read_text())
    manifest_path = repo_root / prior["source_manifest"]
    if dual_sha256(manifest_path) != prior["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the transfer probe")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "ball_coordinate_windows.v1" or manifest["input_kind"] != "mdd_only":
        raise ValueError("Require the pinned coordinate MDD-only manifest")
    config = PlayIntervalConfig(**manifest["config"])
    if config.window_length != 32:
        raise ValueError("The anchor contract requires existing 32-frame windows")
    root = Path(manifest["ball_store"]["directory"])
    if not root.is_relative_to(repo_root):
        raise ValueError("Snapshot path is outside the explicitly supplied source repo")
    for name, expected in manifest["ball_store"]["hashes"].items():
        if dual_sha256(root / name) != expected:
            raise ValueError("Pinned snapshot metadata/index changed")
    paths = [manifest_path, root / "metadata.json", root / "index.npz", *sorted((root / "shards").glob("*.bin"))]
    versions = fingerprint(paths)
    store = BallFrameStore(root)
    # This is evidence about one v1 snapshot, not an alternative v2 teacher reader.
    if store.teacher_masks is not None or store.metadata["schema_version"] != "ball_detection_frames.v1":
        raise ValueError("This survey is pinned to legacy-v1 teacher semantics")
    split_by_group: dict[tuple[str, str], str] = {}
    for clip in store.clips:
        key = clip.source, clip.group_id
        if split_by_group.setdefault(key, clip.split) != clip.split:
            raise ValueError("A source group is shared across data splits")
    by_id = {r["clip"]["clip_id"]: r for r in manifest["clips"]}
    sampling = TemporalSamplingConfig(tuple(manifest["sampling"]["frame_steps"]))
    old_counts: dict[tuple[str, int], int] = defaultdict(int)
    codes_by_clip: dict[int, np.ndarray] = {}
    selections: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    evidence_counts = dict(chat_clips_verified=0, excluded_noted_empty_frames=0)
    annotations = []
    for clip in store.clips:
        states = annotation_states(store, clip)
        codes, evidence = classify(store, clip, states)
        codes_by_clip[clip.index] = codes
        if "annotation_sha256" in evidence:
            evidence_counts["chat_clips_verified"] += 1
            evidence_counts["excluded_noted_empty_frames"] += evidence["excluded_noted_empty_frames"]
            annotations.append([clip.clip_id, evidence["annotation_sha256"], evidence["excluded_noted_empty_frames"]])
        record = by_id.get(clip.clip_id)
        play, regenerated = select_sampled_windows(states, config, sampling)
        if record is None:
            if regenerated:
                raise ValueError("Pinned manifest omitted an eligible clip")
            anchors: list[int] = []
        else:
            if record["windows"] != [asdict(w) for w in regenerated] or record["play"] != [list(p) for p in play]:
                raise ValueError("Pinned play windows disagree with current APIs")
            if record["semantic_sha256"] != clip_semantic_digest(store, clip):
                raise ValueError("Pinned clip identity differs")
            anchors = sorted(w["start"] for w in record["windows"] if w["frame_step"] == 1)
            if len(anchors) != len(set(anchors)):
                raise ValueError("Duplicate anchor")
        centers = [a + 8 for a in anchors]
        occluded = states.occluded > 0
        cases = [("snapshot", 0, list(range(0, clip.frame_count)), 1),
                 ("reference_full32", 32, anchors, 32), ("reference_center16", 32, centers, 16)]
        for context, spans in (("play", list(play)), ("clip", eligible_spans(states, config))):
            previous: set[int] | None = None
            for length in LENGTHS:
                accepted = context_starts(centers, length, spans)
                if previous is not None and not set(accepted) <= previous:
                    raise ValueError("Longer context added targets")
                previous = set(accepted)
                cases.append((context, length, accepted, CENTRAL_LENGTH))
                if context == "play":
                    old_counts[clip.split, length] += len(accepted)
        for context, length, starts, width in cases:
            stats, mask = selection_stats(codes, occluded, starts, width)
            usable, usable_mask = usable_centers(codes, starts, width)
            stats["usable_masked"] = int(np.sum(usable_mask & (codes == 0)))
            stats["usable_unique_frames"] = int(usable_mask.sum())
            selections[clip.split, context, length][clip.index] = dict(
                stats=stats, mask=mask, starts=starts, usable_starts=usable, usable_mask=usable_mask)
        for length in LENGTHS:
            a = selections[clip.split, "play", length][clip.index]
            b = selections[clip.split, "clip", length][clip.index]
            if not set(a["starts"]) <= set(b["starts"]):
                raise ValueError("Play context must be a subset of safe clip context")
        if (clip.index + 1) % 300 == 0:
            print(f"Classified {clip.index + 1}/{len(store.clips)} clips (no images)", flush=True)
    for prior_row in prior["rows"]:
        if (prior_row["frame_step"] == 1 and prior_row["frames"] in LENGTHS
                and old_counts[prior_row["split"], prior_row["frames"]] != prior_row["existing_32_windows_with_symmetric_context"]):
            raise ValueError("Geometry does not reproduce the prior support count")
    if fingerprint(paths) != versions or store._shards or store._video_versions:
        raise ValueError("Inputs changed or image payloads were opened")
    for name, expected in manifest["ball_store"]["hashes"].items():
        if dual_sha256(root / name) != expected:
            raise ValueError("Snapshot changed during survey")
    output.mkdir(parents=True)
    groups: dict[str, Any] = {}
    detail_rows = []
    overall_rows = []
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (split, context, length), selection in sorted(selections.items()):
        clips = [c for c in store.clips if c.split == split]
        dimensions: dict[tuple[str, str], list[ClipRecord]] = defaultdict(list)
        for clip in clips:
            for dim, name in (("all", "all"), ("source", clip.source), ("native_fps", str(Fraction(clip.fps))),
                              ("source_fps", f"{clip.source}|{Fraction(clip.fps)}"), ("group", f"{clip.source}|{clip.group_id}")):
                dimensions[dim, name].append(clip)
        for (dimension, name), subset in sorted(dimensions.items()):
            row: dict[str, Any] = dict(split=split, context=context, frames=length, dimension=dimension, name=name)
            for key in next(iter(selection.values()))["stats"]:
                row[key] = sum(selection[c.index]["stats"][key] for c in subset)
            supported = [c for c in subset if selection[c.index]["stats"]["candidate_blocks"]]
            observed = [c for c in subset if selection[c.index]["stats"]["observed"]]
            negatives = [c for c in subset if selection[c.index]["stats"]["reviewed_empty"] + selection[c.index]["stats"]["out_of_frame"]]
            row.update(clips=len(supported), groups=len({(c.source, c.group_id) for c in supported}),
                       observed_groups=len({(c.source, c.group_id) for c in observed}),
                       negative_groups=len({(c.source, c.group_id) for c in negatives}))
            for label in LABELS[1:]:
                for reference, suffix in (("reference_full32", "full32"), ("reference_center16", "center16"), ("snapshot", "snapshot")):
                    base = selections[split, reference, 0 if reference == "snapshot" else 32]
                    denominator = sum(base[c.index]["stats"][label] for c in subset)
                    # Intersections, not merely ratios of different populations.
                    numerator = sum(int(np.sum(selection[c.index]["mask"] & base[c.index]["mask"] & (codes_by_clip[c.index] == LABELS.index(label)))) for c in subset)
                    row[f"{label}_coverage_{suffix}"] = numerator / denominator if denominator else None
            if dimension == "all":
                masks = {c.index: selection[c.index]["usable_mask"] for c in clips}
                row.update(identity(clips, masks, codes_by_clip))
                overall_rows.append(row)
            elif dimension != "group":
                detail_rows.append(row)
            elif (context in {"play", "clip"} and length in {128, 256}) or context == "reference_full32":
                group_rows[f"groups-{context}-{length}.csv"].append(row)
        if context in {"play", "clip"} and length in {128, 256}:
            lines = []
            for clip in clips:
                starts = selection[clip.index]["usable_starts"]
                if starts:
                    lines.append(json.dumps([clip.index, compress_starts(starts)], separators=(",", ":")))
            (output / f"centers-{context}-{length}-{split}.jsonl").write_text("\n".join(lines) + "\n")
        if context == "snapshot":
            groups[split] = sorted({(c.source, c.group_id) for c in clips})

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "source-fps.csv", detail_rows)
    for name, rows in group_rows.items():
        write_csv(output / name, rows)
    write_json(output / "summary.json", dict(schema="ball_temporal_target_survey.v1", labels=LABELS, rows=overall_rows))
    write_json(output / "inputs.json", dict(
        source_manifest=prior["source_manifest"], source_manifest_sha256=prior["source_manifest_sha256"],
        snapshot=str(root.relative_to(repo_root)), snapshot_hashes=manifest["ball_store"]["hashes"],
        api_sha256={p: dual_sha256(Path(p)) for p in APIS},
        central_length=CENTRAL_LENGTH, frame_step=1, lengths=LENGTHS, config=manifest["config"],
        anchor="existing manifest native-step 32-frame windows; central [start+8,start+24)",
        context="input [center-(T-16)/2,center+16+(T-16)/2); play or hybrid.eligible_spans",
        labels="annotation_states.supervision; hybrid.reviewed_empty_evidence/negative_mask; remaining masked",
        identities="sorted clip_id then frame_index; compact JSONL [clip_id,index,frame_step,exact_fps]; teacher identity appends class code",
        centers="JSONL [snapshot_clip_index, inclusive arithmetic runs [first,last,step]]; excludes fully masked blocks; counts retain geometry candidates separately",
        evidence=evidence_counts, groups=groups,
        checks=dict(prior_geometry_reproduced=True, nested_lengths=True, play_subset_of_clip=True, split_groups_disjoint=True,
                    teacher_partition=True, frozen_clip_semantics=True, source_versions_unchanged=True, image_payloads_opened=0)))
    write_json(output / "annotation-evidence.json", annotations)
    write_json(output / "execution.json", dict(
        inspected_code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        elapsed_seconds=time.perf_counter() - began, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metadata_clips=len(store.clips), frames=len(store), source_files_stat_checked=len(paths),
        note="CPU only; no model construction, image decoding, shard hashing/copying or source writes"))
    print(f"Survey saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    survey(args.repo_root.resolve(), args.source_probe, args.output)
