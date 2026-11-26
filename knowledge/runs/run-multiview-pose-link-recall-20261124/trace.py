"""Trace lost Meiji 3cam dev links from the torso probe to full-body association."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from assignment_probe import (
    DEV_CLIPS,
    hypotheses,
    milp_assignment,
    pair_weights,
    torso_observations,
)
from numpy.typing import NDArray
from replay import _read_human_eval_inputs

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    match_to_labels,
)
from src.tennis_scene.multiview_pose.assignment import assign
from src.tennis_scene.multiview_pose.geometry import (
    compare_rays,
    fit_joint_observations,
    view_rays,
)
from src.tennis_scene.multiview_pose.inputs import PoseInputs
from src.tennis_scene.multiview_pose.reconstruct import _compatible_groups
from src.tennis_scene.multiview_pose.schema import PoseConfig
from src.utils.io import write_json_atomic
from src.utils.schema.player import (
    COCO17_BONE_LENGTH_EDGES,
    COCO17_TORSO_TWIST_JOINTS,
    COCO_KP_NAMES,
)

_TORSO = list(COCO17_TORSO_TWIST_JOINTS)
# Both policies use these settings. The original accepted a group only when
# every joint residual and bone examined by the mixing checks passed.
_BEFORE = PoseConfig()
Unit = tuple[int, int, int]  # camera, label index, and raw row for non-player boxes
Link = tuple[Unit, Unit]


def raw_predictions(inputs: PoseInputs) -> dict[str, CameraPrediction]:
    predictions = {}
    for detection in inputs.detections:
        counts = np.diff(detection.frame_offsets)
        observed = np.arange(counts.max())[:, None] < counts
        boxes = np.zeros((*observed.shape, 4), np.float64)
        boxes.transpose(1, 0, 2)[observed.T] = detection.boxes_xyxy
        predictions[detection.camera_id] = CameraPrediction(
            np.arange(len(boxes), dtype=np.int64),
            boxes,
            observed,
            np.full(observed.shape, -1, dtype=np.int64),
        )
    return predictions


def _units(
    labels: ClipLabels,
    matched: dict[str, NDArray[np.int64]],
    camera_ids: tuple[str, ...],
    frame: int,
) -> tuple[list[dict[int, Unit]], dict[Unit, str]]:
    rows, sides = [], {}
    for camera, name in enumerate(camera_ids):
        annotation = labels.cameras[name]
        indices = annotation.at(frame)
        people = annotation.person_index[indices]
        boxes = annotation.boxes_xyxy[indices]
        bottom = {}
        for p in np.unique(people[people >= 0]):
            candidates = boxes[people == p]
            largest = np.prod(candidates[:, 2:] - candidates[:, :2], axis=1).argmax()
            bottom[int(p)] = candidates[largest, 3]
        players = [y for p, y in bottom.items() if labels.people[p].role == "player"]
        midpoint = (
            np.mean(players) if len(players) == 2 and players[0] != players[1] else None
        )
        mapping = {}
        for row in np.flatnonzero(matched[name][:, frame] >= 0):
            person = int(matched[name][row, frame])
            unit = (
                camera,
                person,
                -1 if labels.people[person].role == "player" else int(row),
            )
            mapping[int(row)] = unit
            sides[unit] = (
                "unknown"
                if midpoint is None
                else ("far" if bottom[person] < midpoint else "near")
            )
        rows.append(mapping)
    return rows, sides


def _links(group, rows: list[dict[int, Unit]]):
    return combinations(
        [mapping[r] for r, mapping in zip(group, rows, strict=True) if r in mapping], 2
    )


def _stat_links(
    links: set[Link], truth: set[Link], sides: dict[Unit, str]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for a, b in links:
        category = (
            "tp"
            if (a, b) in truth
            else (
                "fp_player_player" if a[2] == b[2] == -1 else "fp_involving_non_player"
            )
        )
        side_pair = "+".join(sorted((sides[a], sides[b])))
        counts[f"{category}:{side_pair}"] += 1
    return counts


def _loss_reason(
    a, b, indices, groups, supported, passed, joints, pairs, rows, threshold
):
    if not indices:
        pair = pairs[a[0], b[0]]
        observations = [
            (i, j)
            for i, x in rows[a[0]].items()
            if x == a
            for j, y in rows[b[0]].items()
            if y == b
        ]
        for stage, mask in (
            ("visibility", pair.observed),
            ("angle", pair.angular),
            ("depth", pair.forward),
            ("bounds", pair.physical),
            ("torso_residual", pair.physical & (pair.errors_px <= threshold)),
        ):
            if not any(mask[i, j, _TORSO].sum() >= 3 for i, j in observations):
                return f"candidate_{stage}", None
        raise AssertionError("A compatible pair is missing from the generated groups")
    if any(passed[i] for i in indices):
        return "milp_conflict", None
    supported_indices = [i for i in indices if supported[i]]
    if not supported_indices:
        return "group_support", None
    # Pair-first attribution after ruling out any viable alternative. The
    # same labelled link may occur in a larger hypothesis or on duplicate rows.
    index = min(supported_indices, key=lambda i: int((groups[i] >= 0).sum()))
    residual = bool(joints.mixed_residuals[index].any())
    bone = bool(joints.overlong_bones[index].any())
    if residual and bone:
        return "residual_and_bone", index
    if residual:
        return "residual", index
    if bone:
        return "bone", index
    raise AssertionError("A supported non-mixed group should have been viable")


def trace_clip(dataset: Path, archive: Path, clip: str) -> dict:
    inputs, labels, pins = _read_human_eval_inputs(dataset, archive, clip)
    raw = raw_predictions(inputs)
    matched = match_to_labels(labels, raw, 0.5)
    cameras, size = inputs.cameras, inputs.source.size
    scale = inputs.source.pixel_threshold_scale
    threshold = _BEFORE.reprojection_px * scale
    offsets = [d.frame_offsets for d in inputs.detections]
    rays = [
        view_rays(p.keypoints, c, size, _BEFORE)
        for p, c in zip(inputs.poses, cameras, strict=True)
    ]
    torso = [
        torso_observations(p.keypoints, c, size)
        for p, c in zip(inputs.poses, cameras, strict=True)
    ]
    edges = tuple(combinations(range(len(cameras)), 2))
    lost, gained, joint_flags, bone_flags = Counter(), Counter(), Counter(), Counter()
    probe_strata, original_strata = Counter(), Counter()
    totals = {"positive": 0, "gained": 0}
    for frame in range(inputs.source.num_frames):
        rows, sides = _units(labels, matched, inputs.source.camera_ids, frame)
        truth = {
            (a, b)
            for i, j in edges
            for a in set(rows[i].values())
            for b in set(rows[j].values())
            if a[2] == b[2] == -1 and a[1] == b[1]
        }
        spans = [slice(axis[frame], axis[frame + 1]) for axis in offsets]
        sampled_torso = [
            tuple(a[span] for a in view)
            for span, view in zip(spans, torso, strict=True)
        ]
        weights = {
            (a, b): pair_weights(
                sampled_torso[a], sampled_torso[b], cameras[a], cameras[b], threshold
            )
            for a, b in edges
        }
        probe = set()
        for choice in milp_assignment(hypotheses(weights)):
            view_rows = dict(choice.rows)
            probe.update(
                _links([view_rows.get(v, -1) for v in range(len(cameras))], rows)
            )

        views = tuple(
            ray.slice(span.start, span.stop)
            for ray, span in zip(rays, spans, strict=True)
        )
        pairs = {
            (a, b): compare_rays(
                views[a], views[b], cameras[a], cameras[b], _BEFORE, threshold
            )
            for a, b in edges
        }
        groups = _compatible_groups(pairs, len(cameras))
        joints = fit_joint_observations(groups, views, cameras, pairs, _BEFORE, scale)
        supported = (
            (joints.inliers[..., _TORSO].sum(axis=-1) >= 3) | (groups < 0)
        ).all(axis=1)
        passed = (
            supported
            & ~joints.mixed_residuals.any(axis=1)
            & ~joints.overlong_bones.any(axis=1)
        )
        viable = np.flatnonzero(passed)
        scores = np.zeros(len(groups))
        for left, right in edges:
            residual = (
                np.maximum(joints.errors[:, left], joints.errors[:, right]) / threshold
            )
            scores += np.where(
                joints.inliers[:, left] & joints.inliers[:, right],
                1 - 0.5 * residual**2,
                0,
            ).mean(axis=-1)
        order = assign(
            groups[viable], scores[viable], _BEFORE.ambiguity_relative_gap
        ).selected
        original = {pair for i in viable[order] for pair in _links(groups[i], rows)}

        probe_strata.update(_stat_links(probe, truth, sides))
        original_strata.update(_stat_links(original, truth, sides))
        totals["positive"] += len(truth)
        for a, b in (original - probe) & truth:
            gained["+".join(sorted((sides[a], sides[b])))] += 1
            totals["gained"] += 1
        loss = (probe - original) & truth
        containing: dict[Link, list[int]] = defaultdict(list)
        for index, group in enumerate(groups):
            for link in _links(group, rows):
                if link in loss:
                    containing[link].append(index)
        for a, b in loss:
            reason, index = _loss_reason(
                a,
                b,
                containing[a, b],
                groups,
                supported,
                passed,
                joints,
                pairs,
                rows,
                threshold,
            )
            side_pair = "+".join(sorted((sides[a], sides[b])))
            lost[f"{side_pair}:{reason}"] += 1
            if index is not None:
                joint_flags.update(
                    COCO_KP_NAMES[j]
                    for j in np.flatnonzero(joints.mixed_residuals[index])
                )
                bone_flags.update(
                    "--".join(COCO_KP_NAMES[j] for j in COCO17_BONE_LENGTH_EDGES[i])
                    for i in np.flatnonzero(joints.overlong_bones[index])
                )
    return {
        "pins": pins,
        "positive_links": totals["positive"],
        "probe_links_by_role_and_side": dict(probe_strata),
        "previous_links_by_role_and_side": dict(original_strata),
        "lost_player_links_by_side_and_reason": dict(lost),
        "gained_player_links_by_side": dict(gained),
        "mixed_residual_joint_flags": dict(joint_flags),
        "overlong_bone_flags": dict(bone_flags),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=len(DEV_CLIPS)) as pool:
        jobs = {
            clip: pool.submit(trace_clip, args.dataset, args.archives, clip)
            for clip in DEV_CLIPS
        }
        data = {clip: job.result() for clip, job in jobs.items()}
    write_json_atomic(args.output, data)


if __name__ == "__main__":
    main()
