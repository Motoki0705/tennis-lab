"""CPU-only proposal: common safe spans, paired updates and absence evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.hybrid import eligible_spans
from src.tasks.ball_detection.data.windows.intervals import (
    PlayIntervalConfig,
    mask_intervals,
)
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.tasks.ball_detection.data.windows.temporal import (
    TemporalSamplingConfig,
    select_sampled_windows,
)
from src.utils.checksum import dual_sha256

# Reuse the first run's frozen teacher classification, including Chat notes checks.
PRIOR_RUN = Path(__file__).resolve().parent.parent / "run-ball-temporal-target-survey-20261022"
sys.path.insert(0, str(PRIOR_RUN))
from temporal_target_probe import (  # noqa: E402
    APIS,
    LABELS,
    classify,
    compress_starts,
    fingerprint,
    identity,
    write_json,
)

LENGTHS = (32, 64, 128)
PACK_LENGTH = 128
TRAIN_STRIDE = 16
EDGE_BOUNDS = (0, 1, 8, 16, 32, 64)
EDGE_NAMES = ("0", "1-7", "8-15", "16-31", "32-63", "64+")


def window_starts(start: int, stop: int, length: int, stride: int) -> list[int]:
    """Use real frames, a left-anchored grid and one deduplicated tail backfill."""
    if start < 0 or stop < start or length < 1 or not 1 <= stride <= length:
        raise ValueError("Invalid span, length or stride")
    last = stop - length
    if last < start:
        return []
    starts = list(range(start, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def owner_starts(start: int, stop: int, length: int) -> NDArray[np.int64]:
    """Nearest input-window center; ties go to the earlier start. No label filter."""
    owners: NDArray[np.int64] = np.full(stop - start, -1, dtype=np.int64)
    best: NDArray[np.int64] = np.full(stop - start, np.iinfo(np.int64).max, dtype=np.int64)
    for first in window_starts(start, stop, length, length // 2):
        indices = np.arange(first, first + length)
        distance = np.abs(2 * indices - (2 * first + length - 1))
        section = slice(first - start, first - start + length)
        better = distance < best[section]
        best[section][better] = distance[better]
        owners[section][better] = first
    if np.any(owners < 0):
        raise ValueError("Uncovered evaluation frame")
    return owners


def class_runs(codes: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    """Lossless half-open runs [start, stop, class] for confirmed negatives."""
    runs: list[list[int]] = []
    for code in (2, 3):
        runs.extend([a, b, code] for a, b in mask_intervals(mask & (codes == code)))
    return sorted(runs)


def paired_starts(pack_start: int, length: int) -> list[int]:
    if length not in LENGTHS:
        raise ValueError("Unsupported comparison length")
    return list(range(pack_start, pack_start + PACK_LENGTH, length))


def window_counts(codes: np.ndarray, starts: list[int], length: int) -> NDArray[np.int64]:
    # Prefix counts avoid per-window allocation and never touch RGB payloads.
    prefix = np.vstack([np.r_[0, np.cumsum(codes == i)] for i in range(4)]).T
    indices = np.asarray(starts, dtype=np.int64)
    counts: NDArray[np.int64] = prefix[indices + length] - prefix[indices]
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Empty evidence table")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def survey(source_repo: Path, output: Path) -> None:
    began = time.perf_counter()
    if output.exists() or not output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Require a fresh output directory inside the active checkout")
    pinned = json.loads((PRIOR_RUN / "inputs.json").read_text())
    for path, expected in pinned["api_sha256"].items():
        if dual_sha256(Path(path)) != expected:
            raise ValueError(f"Pinned teacher/window API changed: {path}")
    manifest_path = source_repo / pinned["source_manifest"]
    if dual_sha256(manifest_path) != pinned["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the previous survey")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "ball_coordinate_windows.v1" or manifest["input_kind"] != "mdd_only":
        raise ValueError("Require the pinned native-step coordinate manifest")
    root = Path(manifest["ball_store"]["directory"])
    if not root.is_relative_to(source_repo):
        raise ValueError("Snapshot is outside the explicitly provided source repo")
    for path, expected in manifest["ball_store"]["hashes"].items():
        if dual_sha256(root / path) != expected:
            raise ValueError("Pinned snapshot index/metadata changed")
    store = BallFrameStore(root)
    if store.teacher_masks is not None or store.metadata["schema_version"] != "ball_detection_frames.v1":
        raise ValueError("This evidence uses the fixed legacy-v1 teacher contract")
    # Original TrackNet/Meiji files are not required by the normalized contract.
    # Only Chat originals consulted by reviewed_empty_evidence are stat-checked.
    annotation_paths = {c["clip_id"]: c.get("annotation_path") for c in store.metadata["clips"]}
    paths = [manifest_path, root / "index.npz", root / "metadata.json",
             *sorted((root / "shards").glob("*.bin"))]
    versions = fingerprint(paths)
    config = PlayIntervalConfig(**manifest["config"])
    if config.window_length != 32:
        raise ValueError("Unexpected baseline window length")
    sampling = TemporalSamplingConfig(tuple(manifest["sampling"]["frame_steps"]))
    records = {r["clip"]["clip_id"]: r for r in manifest["clips"]}
    group_splits: dict[tuple[str, str], str] = {}
    data: dict[int, dict[str, Any]] = {}
    compact: dict[str, list[Any]] = defaultdict(list)
    budget: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    budgets_extra: dict[tuple[str, int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    edge_counts: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    eval_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    owner_hashes = {(s, t): hashlib.sha256() for s in ("train", "val", "test") for t in LENGTHS}
    annotation_evidence = []
    catalog = []
    for clip in sorted(store.clips, key=lambda c: c.clip_id):
        group = (clip.source, clip.group_id)
        if group_splits.setdefault(group, clip.split) != clip.split:
            raise ValueError("A source group crosses split boundaries")
        states = annotation_states(store, clip)
        if clip.source == "chat_annotation" and np.any(states.reviewed & states.target & (states.count == 0)):
            if not annotation_paths[clip.clip_id]:
                raise ValueError("Chat absence evidence requires the original annotation path")
            path = Path(annotation_paths[clip.clip_id])
            paths.append(path)
            versions.update(fingerprint([path]))
        codes, evidence = classify(store, clip, states)
        if "annotation_sha256" in evidence:
            annotation_evidence.append([clip.clip_id, evidence["annotation_sha256"], evidence["excluded_noted_empty_frames"]])
        play, windows = select_sampled_windows(states, config, sampling)
        record = records.get(clip.clip_id)
        if record is None:
            if windows:
                raise ValueError("Manifest omits an eligible baseline clip")
        elif (record["windows"] != [asdict(w) for w in windows]
              or record["play"] != [list(p) for p in play]
              or record["semantic_sha256"] != clip_semantic_digest(store, clip)):
            raise ValueError("Baseline manifest disagrees with its pinned source")
        reference = np.zeros(clip.frame_count, dtype=np.bool_)
        for w in windows:
            if w.frame_step == 1:
                reference[w.start:w.start + 32] = True
        spans = eligible_spans(states, config)
        common_spans = [(a, b) for a, b in spans if b - a >= PACK_LENGTH]
        common = np.zeros(clip.frame_count, dtype=np.bool_)
        pack_starts: list[int] = []
        for a, b in common_spans:
            common[a:b] = True
            starts = window_starts(a, b, PACK_LENGTH, TRAIN_STRIDE)
            counts = window_counts(codes, starts, PACK_LENGTH)
            accepted = counts[:, 1:].sum(axis=1) > 0
            pack_starts.extend(s for s, keep in zip(starts, accepted, strict=True) if keep)
            # Every T partitions precisely the same accepted 128-frame samples.
            for length in LENGTHS:
                parts = [s for p in np.asarray(starts)[accepted] for s in paired_starts(int(p), length)]
                part_counts = window_counts(codes, parts, length)
                paired = part_counts.reshape(-1, PACK_LENGTH // length, 4).sum(axis=1)
                if not np.array_equal(paired, counts[accepted]):
                    raise ValueError("Paired updates changed frame/teacher multiplicities")
                key = (clip.split, length, "paired_known1")
                budget[key].append(paired)
                budgets_extra[key]["candidate_windows"] += len(starts) * (PACK_LENGTH // length)
                budgets_extra[key]["accepted_windows"] += len(parts)
                budgets_extra[key]["zero_teacher_child_windows"] += int((part_counts[:, 1:].sum(axis=1) == 0).sum())
                budgets_extra[key]["negative_only_packs"] += int(((paired[:, 1] == 0) & (paired[:, 2:].sum(axis=1) > 0)).sum())
                # A simple per-T sampler is measured as a comparison, not adopted.
                independent = window_counts(codes, window_starts(a, b, length, TRAIN_STRIDE), length)
                for rule, keep in (("independent_known1", independent[:, 1:].sum(axis=1) > 0),
                                   ("independent_observed25pct", independent[:, 1] * 4 >= length)):
                    ikey = (clip.split, length, rule)
                    budget[ikey].append(independent[keep])
                    budgets_extra[ikey]["candidate_windows"] += len(independent)
                    budgets_extra[ikey]["accepted_windows"] += int(keep.sum())
                    budgets_extra[ikey]["zero_teacher_child_windows"] += 0
                    budgets_extra[ikey]["negative_only_packs"] += int(((independent[keep, 1] == 0) & (independent[keep, 2:].sum(axis=1) > 0)).sum())
                owners = owner_starts(a, b, length)
                indices = np.arange(a, b)
                labeled = codes[a:b] > 0
                owner_hashes[clip.split, length].update((json.dumps([clip.clip_id, a, b, owners.tolist()], separators=(",", ":")) + "\n").encode())
                ec = eval_counts[clip.split, length]
                ec["windows"] += len(window_starts(a, b, length, length // 2))
                ec["assigned_frames"] += b - a
                ec["assigned_labeled_frames"] += int(labeled.sum())
                ec["observed_at_window_start"] += int(((indices == owners) & (codes[a:b] == 1)).sum())
                ec["negative_at_window_start"] += int(((indices == owners) & (codes[a:b] >= 2)).sum())
            distance = np.minimum(np.arange(b - a), np.arange(b - a)[::-1])
            bins = np.searchsorted(EDGE_BOUNDS, distance, side="right") - 1
            for i, name in enumerate(EDGE_NAMES):
                edge_counts[clip.split, name] += np.bincount(codes[a:b][bins == i], minlength=4)
        negative = np.zeros(clip.frame_count, dtype=np.bool_)
        for a, b in spans:
            if b - a >= PACK_LENGTH:
                negative[a:b] = codes[a:b] >= 2
        # The full geometric common region includes every evaluable absence frame.
        if not np.array_equal(negative, common & (codes >= 2)):
            raise ValueError("Absence membership changed")
        masks = dict(snapshot=np.ones(clip.frame_count, dtype=np.bool_), reference32=reference,
                     common=common, absence=negative, absence_overlap=negative & common,
                     absence_exclusive=negative & ~common)
        supported = np.zeros(clip.frame_count, dtype=np.bool_)
        for first in pack_starts:
            supported[first:first + PACK_LENGTH] = True
        if not np.array_equal(common & (codes > 0), supported & (codes > 0)):
            raise ValueError("Training acceptance lost common labeled frames")
        data[clip.index] = dict(codes=codes, masks=masks, spans=common_spans, occluded=states.occluded > 0)
        fps = str(Fraction(clip.fps))
        catalog.append(dict(clip_index=clip.index, clip_id=clip.clip_id, split=clip.split, source=clip.source,
                            group_id=clip.group_id, native_fps=fps, frames=clip.frame_count))
        if common_spans:
            compact[f"common-intervals-{clip.split}.jsonl"].append([clip.index, common_spans])
        if pack_starts:
            compact[f"train-packs-{clip.split}.jsonl"].append([clip.index, compress_starts(pack_starts)])
        if negative.any():
            runs = [[a + x, a + y, code] for a, b in common_spans
                    for x, y, code in class_runs(codes[a:b], negative[a:b])]
            compact[f"absence-targets-{clip.split}.jsonl"].append([clip.index, runs])
        if len(data) % 300 == 0:
            print(f"Classified {len(data)}/{len(store.clips)} clips (no images)", flush=True)

    overall, details = [], []
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    codes_by_clip = {i: r["codes"] for i, r in data.items()}
    for split in ("train", "val", "test"):
        clips = [c for c in store.clips if c.split == split]
        dimensions: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for c in clips:
            for dim, name in (("all", "all"), ("source", c.source), ("native_fps", str(Fraction(c.fps))),
                              ("source_fps", f"{c.source}|{Fraction(c.fps)}"), ("group", f"{c.source}|{c.group_id}")):
                dimensions[dim, name].append(c)
        for scope in ("snapshot", "reference32", "common", "absence", "absence_overlap", "absence_exclusive"):
            for (dimension, name), subset in sorted(dimensions.items()):
                row: dict[str, Any] = dict(split=split, scope=scope, dimension=dimension, name=name)
                counts = np.zeros(4, dtype=np.int64)
                active, observed_groups, negative_groups = [], set(), set()
                intervals = occluded = 0
                denominators: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
                intersections: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
                for c in subset:
                    d = data[c.index]
                    mask = d["masks"][scope]
                    cc = np.bincount(d["codes"][mask], minlength=4)
                    counts += cc
                    baseline = d["masks"]["reference32"]
                    denominators += np.bincount(d["codes"][baseline], minlength=4)
                    intersections += np.bincount(d["codes"][mask & baseline], minlength=4)
                    if mask.any():
                        active.append(c)
                        intervals += len(mask_intervals(mask)) if scope != "common" else len(d["spans"])
                    if cc[1]:
                        observed_groups.add((c.source, c.group_id))
                    if cc[2:].sum():
                        negative_groups.add((c.source, c.group_id))
                    occluded += int((mask & (d["codes"] == 1) & d["occluded"]).sum())
                row.update(frames=int(counts.sum()), intervals=intervals, clips=len(active),
                           groups=len({(c.source, c.group_id) for c in active}), observed_groups=len(observed_groups),
                           negative_groups=len(negative_groups), observed_occluded=occluded)
                row.update({name: int(counts[i]) for i, name in enumerate(LABELS)})
                for i, label in enumerate(LABELS):
                    row[f"{label}_reference32"] = int(denominators[i])
                    row[f"{label}_intersection32"] = int(intersections[i])
                    row[f"{label}_added_outside32"] = int(counts[i] - intersections[i])
                    row[f"{label}_coverage32"] = float(intersections[i] / denominators[i]) if denominators[i] else None
                if dimension == "all":
                    row.update(identity(clips, {c.index: data[c.index]["masks"][scope] for c in clips}, codes_by_clip))
                    overall.append(row)
                elif dimension == "group":
                    group_rows[scope].append(row)
                else:
                    details.append(row)
    budget_rows = []
    for (split, length, rule), count_parts in sorted(budget.items()):
        values = np.concatenate(count_parts, axis=0)
        unit_frames = PACK_LENGTH if rule.startswith("paired") else length
        expected = values.mean(axis=0) * PACK_LENGTH / unit_frames
        row = dict(split=split, frames_per_window=length, rule=rule, frames_per_update=PACK_LENGTH,
                   windows_per_update=PACK_LENGTH // length, sampling_units=len(values),
                   **budgets_extra[split, length, rule])
        row.update({f"expected_{name}": float(expected[i]) for i, name in enumerate(LABELS)})
        row["expected_negative"] = float(expected[2:].sum())
        row["negative_only_unit_fraction"] = float(((values[:, 1] == 0) & (values[:, 2:].sum(axis=1) > 0)).mean())
        budget_rows.append(row)
    edge_rows = []
    for (split, name), counts in sorted(edge_counts.items()):
        edge_rows.append(dict(split=split, distance_from_interval_edge_frames=name,
                              frames=int(counts.sum()), **{n: int(counts[i]) for i, n in enumerate(LABELS)}))
    eval_rows = [dict(split=s, frames_per_window=t, **v, owner_sha256=owner_hashes[s, t].hexdigest())
                 for (s, t), v in sorted(eval_counts.items())]
    # Verify teacher classification against the independent first run's snapshot totals.
    previous = json.loads((PRIOR_RUN / "summary.json").read_text())
    for row in overall:
        if row["scope"] in {"snapshot", "reference32"}:
            old_scope = "snapshot" if row["scope"] == "snapshot" else "reference_full32"
            old = next(r for r in previous["rows"] if r["split"] == row["split"] and r["context"] == old_scope)
            if any(row[k] != old[k] for k in LABELS):
                raise ValueError("Baseline unique teacher counts do not reproduce the first run")
    if fingerprint(paths) != versions or store._shards or store._video_versions:
        raise ValueError("Inputs changed or image payloads were opened")
    for path, expected in manifest["ball_store"]["hashes"].items():
        if dual_sha256(root / path) != expected:
            raise ValueError("Snapshot changed during the survey")
    output.mkdir(parents=True)
    for name, rows in compact.items():
        write_jsonl(output / name, rows)
    write_csv(output / "clip-catalog.csv", catalog)
    write_csv(output / "source-fps.csv", details)
    for scope, rows in group_rows.items():
        write_csv(output / f"groups-{scope}.csv", rows)
    write_csv(output / "budget.csv", budget_rows)
    write_csv(output / "edge-support.csv", edge_rows)
    write_csv(output / "evaluation-support.csv", eval_rows)
    write_json(output / "summary.json", dict(schema="ball_all_frame_survey.v1", labels=LABELS, rows=overall))
    write_json(output / "annotation-evidence.json", annotation_evidence)
    write_json(output / "inputs.json", dict(
        source_manifest=pinned["source_manifest"], source_manifest_sha256=pinned["source_manifest_sha256"],
        snapshot=str(root.relative_to(source_repo)), snapshot_hashes=manifest["ball_store"]["hashes"],
        api_sha256={p: dual_sha256(Path(p)) for p in APIS},
        prior_probe_sha256=dual_sha256(PRIOR_RUN / "temporal_target_probe.py"),
        clip_count=len(store.clips), frame_count=len(store), lengths=LENGTHS, frame_step=1,
        config=manifest["config"], chat_clips_verified=len(annotation_evidence),
        excluded_noted_empty_frames=sum(r[2] for r in annotation_evidence),
        checks=dict(source_versions_unchanged=True, split_groups_disjoint=True, baseline_counts_reproduced=True,
                    accepted_packs_cover_all_common_teachers=True, paired_frame_teacher_multiplicities_equal=True,
                    evaluation_covers_every_common_frame=True, image_payloads_opened=0)))
    write_json(output / "assignment-rules.json", dict(
        schema="ball_all_frame_assignment_proposal.v1", status="awaiting_human_agreement",
        common_intervals="all hybrid.eligible_spans of length >=128; no play/label-density filter",
        interval_encoding="JSONL [snapshot_clip_index, [[start,stop_exclusive],...]]; never merge adjacent PTS-separated intervals",
        pack_encoding="JSONL [snapshot_clip_index, [[first_start,last_start_inclusive,stride],...]]",
        absence_encoding="JSONL [snapshot_clip_index, [[start,stop_exclusive,class_code],...]]; 2=reviewed_empty,3=out_of_frame",
        training=dict(pack_length=128, pack_start_stride=16, tail="append stop-128 once",
                      accept="at least one known teacher in the shared 128-frame pack; min observed=0",
                      child_starts="pack_start + j*T, j=0..128/T-1; no child-level rejection",
                      sampling="uniform over accepted train packs, with replacement; same seeded pack stream for all T/models",
                      update="one shared pack for F=128, q packs for F=128*q; every child is forwarded",
                      zero_teacher_child="zero contribution, normalized by the whole update's known teacher count",
                      absence_mixture="not set; auxiliary absence targets are not an additional training sampler"),
        evaluation=dict(start_stride="T//2", tail="append stop-T once; no padding or cropping",
                        owner="argmin over covering windows of (abs(2*frame_index-(2*start+T-1)),start)",
                        key="clip_id,frame_index,frame_step=1,exact_native_fps", score="all labeled frames once, never gate observed",
                        edge_distance="min(frame_index-interval_start,interval_stop-1-frame_index)",
                        edge_bins=list(EDGE_NAMES), absence="confirmed negatives in any safe span length>=128; all overlap common",
                        overlap="report common and absence scopes separately; union counts each frame once; exclusive has support 0")))
    write_json(output / "execution.json", dict(
        inspected_code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        elapsed_seconds=time.perf_counter() - began, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_files_stat_checked=len(paths), metadata_clips=len(store.clips), frames=len(store),
        note="CPU only; metadata/index/annotation reads and file stats; no RGB opens, decoding, GPU, source writes or links"))
    for path in output.iterdir():
        if path.stat().st_size > 256 * 1024:
            raise ValueError(f"Evidence file exceeds the knowledge storage limit: {path.name}")
    print(f"Survey saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    survey(args.source_repo.resolve(), args.output)
