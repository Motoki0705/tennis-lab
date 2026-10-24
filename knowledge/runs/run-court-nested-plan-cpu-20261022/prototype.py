"""CPU-only experiment; numeric choices are probes, not production policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml
from scipy.spatial import cKDTree

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    _points_local,
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _intrinsics_from_hfov,
    _look_at_opencv,
    assign_group_disjoint_splits,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sfm_bounds import (
    bound_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    resolve_target_court,
    target_court_policy_for_trajectory,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_view import (
    AmbiguousCameraRelativeNearFarError,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    project_court_semantics_v3,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCurveMode,
    OrbitSamplingPolicy,
    OrbitShape,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
    read_court_rgb,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    validate_standard_scene_export,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.schema.court import COURT_SKELETON

SEED = 20261022
EVAL_SEED = 77123
BASE_GROUPS = 16
BASE_TARGET = 24
STEP_M = 6.0
GUARD_M = 0.5
DENSE_COUNT = 4096


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def rng_for(*parts: Any) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(digest(parts), 16)))


@dataclass
class Scene:
    sid: str
    alignment: Any
    cameras: tuple[SceneCamera, ...]
    config: CourtDatasetConfiguration
    hashes: dict[str, str]
    compatibility: list[str]


def load_scene(root: Path, sid: str) -> Scene:
    directory = root / sid
    paths = [
        "alignment/alignment.json",
        "reconstruction/export/scene.json",
        "reconstruction/export/cameras.json",
        "resolved-config.yaml",
        "datasets/court/dataset.json",
        "datasets/court/samples/index.npz",
    ]
    hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in paths
    }
    alignment = AlignmentResult.from_dict(
        json.loads((directory / paths[0]).read_text())
    )
    export = validate_standard_scene_export(directory / paths[1])
    cameras = tuple(
        replace(
            camera,
            camera_to_scene=alignment.metric_adapter.metric_from_nht_camera(
                camera.camera_to_scene
            ),
        )
        for camera in export.cameras
    )
    raw = yaml.safe_load((directory / paths[3]).read_text())["dataset"]["court"]
    compatibility = []
    if sid == "B01":
        if "sfm_complex_center_on_hull" in raw["trajectory"]:
            raise ValueError("B01 probe expected the recorded legacy missing key")
        raw["trajectory"]["sfm_complex_center_on_hull"] = False
        compatibility.append(
            "B01: explicit experimental addition sfm_complex_center_on_hull=false; source unchanged"
        )
    return Scene(
        sid,
        alignment,
        cameras,
        CourtDatasetConfiguration.from_mapping(raw),
        hashes,
        compatibility,
    )


@dataclass
class Track:
    scene: Scene
    spec: OrbitTrajectorySpec
    center: OrbitCenter
    family: str
    split: str
    seed: int
    hfov: float
    look_height: float
    dense_theta: Any
    cumulative: Any
    length: float
    n0: int

    def points(self, level: int) -> dict[str, np.ndarray]:
        if level < 0:
            raise ValueError("density level must be nonnegative")
        result = {}
        for j in range(self.n0 * 2**level):
            fraction = Fraction(j, self.n0 * 2**level)
            key = f"{fraction.numerator}/{fraction.denominator}"
            theta = np.interp(
                float(fraction) * self.length, self.cumulative, self.dense_theta
            )
            result[key] = self.center.scene_from_center.apply(
                _points_local(self.spec, np.array([theta]))
            )[0]
        return result

    def dense(self) -> np.ndarray:
        # Uniform in arc length: every continuous point is within length/(2*N).
        fractions = np.arange(1024, dtype=np.float64) / 1024
        theta = np.interp(fractions * self.length, self.cumulative, self.dense_theta)
        points: np.ndarray = np.asarray(
            self.center.scene_from_center.apply(_points_local(self.spec, theta)),
            dtype=np.float64,
        )
        return points


def make_track(
    scene: Scene, ordinal: int, *, seed: int = SEED, split: str = "train"
) -> Track:
    # Both centre kinds and four shapes have balanced marginals in each 16 block.
    strata = ordinal % 16
    random = rng_for("trajectory-probe-v1", scene.sid, seed, ordinal)
    centers = derive_orbit_centers(
        scene.cameras,
        scene.alignment.layout,
        use_court_centroid=True,
        use_captured_hull_centroid=True,
    )
    if strata // 8 == 0:
        center = centers[0]
    else:
        center = centers[1 + (ordinal // 16 + strata) % (len(centers) - 1)]
    shape = list(OrbitShape)[strata % 4]
    sinusoidal = bool((strata // 4) % 2)
    family = digest(["family-probe-v1", scene.sid, seed, ordinal])
    spec = OrbitTrajectorySpec(
        trajectory_id="trajectory-" + family,
        trajectory_group_id="family-" + family,
        shape=shape,
        center_kind=center.center_kind,
        center_court_instance_id=center.court_instance_id,
        base_radius_m=center.base_radius_m,
        radius_scale=float(random.uniform(0.65, 0.95)),
        axis_ratio=1.0
        if shape is OrbitShape.CIRCLE
        else float(random.uniform(0.35, 0.8)),
        orientation_radians=0.0
        if shape is OrbitShape.CIRCLE
        else float(random.uniform(0, math.pi)),
        base_height_m=float(random.uniform(1.5, 3.0)),
        vertical_amplitude_m=float(random.uniform(0.05, 0.5)) if sinusoidal else 0.0,
        vertical_cycles=1 if sinusoidal else 0,
        vertical_phase_radians=float(random.uniform(0, 2 * math.pi))
        if sinusoidal
        else 0.0,
        curve_mode=OrbitCurveMode.SINUSOIDAL_HEIGHT
        if sinusoidal
        else OrbitCurveMode.PLANAR,
    )
    spec = bound_trajectory_candidates(
        [spec],
        centers=centers,
        cameras=scene.cameras,
        margin_m=0.5,
        expansion_percent=0.0,
    )[0]
    theta = np.linspace(0, 2 * math.pi, DENSE_COUNT + 1)
    local = _points_local(spec, theta)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
    length = float(cumulative[-1])
    n0 = max(8, 4 * math.ceil(math.ceil(length / STEP_M) / 4))
    return Track(
        scene,
        spec,
        center,
        family,
        split,
        seed,
        [75.0, 92.5, 110.0][ordinal % 3],
        [0.0, 1.5][ordinal % 2],
        theta,
        cumulative,
        length,
        n0,
    )


def separated(a: Track, b: Track) -> bool:
    # Conservative position-only guard; stronger than a pose + FOV predicate.
    distance = float(cKDTree(a.dense()).query(b.dense())[0].min())
    interpolation_allowance = (a.length + b.length) / (2 * 1024)
    return distance > GUARD_M + interpolation_allowance


def freeze_evaluation(scene: Scene) -> list[Track]:
    result: list[Track] = []
    for split in ["validation", "test"]:
        for ordinal in range(128):
            track = make_track(scene, ordinal, seed=EVAL_SEED, split=split)
            if all(separated(track, other) for other in result):
                result.append(track)
                break
        else:
            raise ValueError("evaluation guard exhausted its fixed probe budget")
    return result


def plan(track: Track, level: int) -> list[dict[str, Any]]:
    scene = track.scene
    result = []
    width, height = scene.cameras[0].width, scene.cameras[0].height
    k = _intrinsics_from_hfov(width=width, height=height, hfov_degrees=track.hfov)
    for fraction, position in track.points(level).items():
        sid = "point-" + digest([track.family, "view-0", fraction])
        target = resolve_target_court(
            policy=target_court_policy_for_trajectory(track.spec),
            camera_center_scene_m=position,
            layout=scene.alignment.layout,
            selection_seed=track.seed,
        )
        random = rng_for("look-at-probe-v1", track.seed, sid)
        radius, angle = (
            math.sqrt(float(random.random())),
            2 * math.pi * float(random.random()),
        )
        court = scene.alignment.layout.court(target.binding.court_instance_id)
        look_at = court.scene_from_court.apply(
            np.array(
                [
                    [
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        track.look_height,
                    ]
                ]
            )
        )[0]
        matrix = _look_at_opencv(
            position,
            look_at,
            vertical_scene=track.center.scene_from_center.matrix()[:3, 2],
        )
        camera = SceneCamera(
            sid,
            0,
            width,
            height,
            k,
            RigidTransform.from_matrix(matrix),
            f"generated/{sid}.png",
        )
        reason = "ok"
        projection = None
        try:
            projection = project_court_semantics_v3(
                camera, scene.alignment.layout
            ).to_dict()
        except AmbiguousCameraRelativeNearFarError:
            reason = "ambiguous_camera_relative_near_far"
        visible, all_visible = 0, 0
        if projection is not None:
            for projected in projection["courts"]:
                count = sum(
                    point["in_frame"]
                    for cls in projected["classes"]
                    for point in cls["points"]
                )
                all_visible += count
                if projected["court_instance_id"] == target.binding.court_instance_id:
                    visible = count
            if all_visible < 4:
                reason = "insufficient_pre_render_semantic_coverage"
            elif visible < 4:
                reason = "insufficient_target_coverage_probe"
        result.append(
            {
                "id": sid,
                "family": track.family,
                "split": track.split,
                "fraction": fraction,
                "K": list(k),
                "pose": matrix.tolist(),
                "look_at": look_at.tolist(),
                "target": target.binding.court_instance_id,
                "projection": projection,
                "visible_target": visible,
                "visible_all": all_visible,
                "reason": reason,
            }
        )
    return result


def eligible(point: dict[str, Any], *, rate: float = 0.8) -> bool:
    # Synthetic independent QA loss, explicitly not a measured renderer rate.
    return (
        point["reason"] == "ok"
        and int(digest(["synthetic-qa-loss", point["id"]])[:16], 16) / 2**64 < rate
    )


def take_balanced(points: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        if eligible(point):
            groups.setdefault(point["family"], []).append(point)
    for group in groups.values():
        group.sort(key=lambda point: digest(["selection-probe", point["id"]]))
    ordered = [
        group[i]
        for i in range(max(map(len, groups.values()), default=0))
        for _, group in sorted(groups.items())
        if i < len(group)
    ]
    if len(ordered) < count:
        raise ValueError(
            f"insufficient accepted candidates: target={count}, available={len(ordered)}"
        )
    return ordered[:count]


def compare(common: list[dict[str, Any]], other: list[dict[str, Any]]) -> int:
    by_id = {point["id"]: point for point in other}
    for point in common:
        if point != by_id[point["id"]]:
            raise AssertionError(f"common geometry changed: {point['id']}")
    return len(common)


def experiment(
    scene: Scene,
) -> tuple[dict[str, Any], list[Track], list[dict[str, Any]]]:
    evaluation = freeze_evaluation(scene)
    eval_points = [point for track in evaluation for point in plan(track, 0)]
    eval_hash = digest(eval_points)
    tracks: list[Track] = []
    excluded = 0
    empty_baseline_families = 0
    for ordinal in range(2048):
        candidate = make_track(scene, ordinal)
        if all(separated(candidate, track) for track in evaluation):
            # Freeze A2's actual trajectory set at the baseline. This simulates
            # the QA preflight that production needs before freezing its plan.
            if len(tracks) < BASE_GROUPS and not any(
                eligible(point) for point in plan(candidate, 0)
            ):
                empty_baseline_families += 1
                continue
            tracks.append(candidate)
        else:
            excluded += 1
        if len(tracks) == BASE_GROUPS * 16:
            break
    else:
        raise ValueError("train guard exhausted its fixed probe budget")
    base_points = [point for track in tracks[:BASE_GROUPS] for point in plan(track, 0)]
    common_checks = {
        "more_tracks": compare(
            base_points, [p for t in tracks[: BASE_GROUPS * 2] for p in plan(t, 0)]
        ),
        "more_density": compare(
            base_points, [p for t in tracks[:BASE_GROUPS] for p in plan(t, 1)]
        ),
    }
    eval_after = [
        point for track in freeze_evaluation(scene) for point in plan(track, 0)
    ]
    if digest(eval_after) != eval_hash:
        raise AssertionError("evaluation changed")
    # A changed training seed cannot enter the independent evaluation namespace.
    changed_train = make_track(scene, 0, seed=SEED + 1)
    if changed_train.family in {t.family for t in evaluation}:
        raise AssertionError("train and evaluation family collision")
    a1, a2, rows, all_candidates = [], [], [], []
    previous_n, previous_a2_ids = 0, set()
    for level in range(5):
        n = BASE_GROUPS * 2**level
        block = [p for t in tracks[previous_n:n] for p in plan(t, 0)]
        delta = BASE_TARGET if level == 0 else BASE_TARGET * 2 ** (level - 1)
        a1.extend(take_balanced(block, delta))
        dense = (
            base_points
            if level == 0
            else [p for t in tracks[:BASE_GROUPS] for p in plan(t, level)]
        )
        compare(base_points, dense)
        new = [p for p in dense if p["id"] not in previous_a2_ids]
        a2.extend(take_balanced(new, delta))
        if {p["family"] for p in a2} != {t.family for t in tracks[:BASE_GROUPS]}:
            raise AssertionError("A2 actual trajectory set changed")
        if level == 0 and a1 != a2:
            raise AssertionError("shared baseline differs")
        rows.append(
            {
                "level": level,
                "target_per_scene": BASE_TARGET * 2**level,
                "a1_planned_groups": n,
                "a1_candidates": len(block),
                "a1_eligible_new": sum(map(eligible, block)),
                "a1_selected": len(a1),
                "a1_used_groups": len({p["family"] for p in a1}),
                "a2_candidates": len(dense),
                "a2_eligible_new": sum(map(eligible, new)),
                "a2_selected": len(a2),
                "a2_used_groups": len({p["family"] for p in a2}),
            }
        )
        all_candidates.extend(block)
        previous_n, previous_a2_ids = n, {p["id"] for p in dense}
    # In the scaled B experiment total=384, thus each scene needs 384/192/128/96.
    b = {str(s): [p["id"] for p in a1[: BASE_TARGET * 16 // s]] for s in range(1, 5)}
    if b["4"] != [p["id"] for p in a1[: BASE_TARGET * 4]]:
        raise AssertionError("B four-scene condition differs from A1 rung 2")
    try:
        take_balanced([], 1)
    except ValueError as error:
        shortage = str(error)
    else:
        raise AssertionError("shortage did not fail")
    # Current implementation counterexamples use a non-round, real-scene path.
    old_policy = OrbitSamplingPolicy.from_configuration(scene.config.sampling)
    track = tracks[0]
    coarse = sample_uniform_arc_length(track.spec, track.center, old_policy)
    fine = sample_uniform_arc_length(
        track.spec,
        track.center,
        replace(old_policy, max_arc_step_m=old_policy.max_arc_step_m / 2),
    )
    for candidate in tracks:
        candidate_coarse = sample_uniform_arc_length(
            candidate.spec, candidate.center, old_policy
        )
        candidate_fine = sample_uniform_arc_length(
            candidate.spec,
            candidate.center,
            replace(old_policy, max_arc_step_m=old_policy.max_arc_step_m / 2),
        )
        if len(candidate_fine.points_scene_m) != 2 * len(
            candidate_coarse.points_scene_m
        ):
            coarse, fine = candidate_coarse, candidate_fine
            break
    common_old = sum(
        np.any(np.linalg.norm(fine.points_scene_m - p, axis=1) < 1e-9)
        for p in coarse.points_scene_m
    )
    old_split_small = assign_group_disjoint_splits(
        [t.family for t in tracks[:BASE_GROUPS]], fractions=(0.8, 0.1, 0.1), seed=695
    )
    old_split_large = assign_group_disjoint_splits(
        [t.family for t in tracks[: BASE_GROUPS * 2]],
        fractions=(0.8, 0.1, 0.1),
        seed=695,
    )
    inventory_centers = derive_orbit_centers(
        scene.cameras,
        scene.alignment.layout,
        use_court_centroid=True,
        use_captured_hull_centroid=scene.config.trajectory.sfm_complex_center_on_hull,
    )
    inventory = generate_trajectory_candidates(
        scene.config.trajectory,
        inventory_centers,
        seed=695,
        stable_field_order=old_policy.stable_field_order,
    )
    metric_centers = np.stack(
        [c.camera_to_scene.matrix()[:3, 3] for c in scene.cameras]
    )
    local_capture = tracks[0].center.scene_from_center.inverse().apply(metric_centers)
    lengths = [t.length for t in tracks]
    drift_counts = [
        {
            "level": level,
            "a1_candidates": sum(t.n0 for t in tracks[: BASE_GROUPS * 2**level]),
            "a2_candidates": sum(t.n0 for t in tracks[:BASE_GROUPS]) * 2**level,
        }
        for level in range(5)
    ]
    acceptance = {
        str(rate): sum(eligible(p, rate=rate) for p in all_candidates)
        for rate in [1.0, 0.9, 0.7]
    }
    stats = {
        "scene": scene.sid,
        "input_sha256": scene.hashes,
        "compatibility": scene.compatibility,
        "alignment_schema": scene.alignment.to_dict()["schema"],
        "captured_cameras": len(scene.cameras),
        "courts": len(scene.alignment.layout.courts),
        "saved_hfov": list(scene.config.view.hfov_degrees),
        "saved_hull_center": scene.config.trajectory.sfm_complex_center_on_hull,
        "saved_expansion_percent": scene.config.trajectory.sfm_boundary_expansion_percent,
        "saved_candidate_inventory": len(inventory),
        "capture_local_xyz_min": local_capture.min(axis=0).tolist(),
        "capture_local_xyz_max": local_capture.max(axis=0).tolist(),
        "length_min_m": min(lengths),
        "length_median_m": float(np.median(lengths)),
        "length_max_m": max(lengths),
        "guard_excluded_families": excluded,
        "empty_baseline_families": empty_baseline_families,
        "guard_kept_families": len(tracks),
        "train_shape_counts": dict(Counter(t.spec.shape.value for t in tracks)),
        "train_center_counts": dict(Counter(t.spec.center_kind.value for t in tracks)),
        "evaluation_hash": eval_hash,
        "evaluation_points": len(eval_points),
        "evaluation_families": [t.family for t in evaluation],
        "evaluation_guard": "position only, dense 1024 point bound; approximate arc-length error not certified",
        "common_geometry_equal": common_checks,
        "counts": rows,
        "length_count_drift": drift_counts,
        "geometry_reasons": dict(Counter(p["reason"] for p in all_candidates)),
        "simulated_acceptance_counts": acceptance,
        "shortage": shortage,
        "selected_union_count": len({p["id"] for p in a1 + a2}),
        "candidate_union_count": len({p["id"] for p in all_candidates + dense}),
        "b_count_per_scene": {s: len(ids) for s, ids in b.items()},
        "b_subset_of_a1_max": all(
            set(ids).issubset({p["id"] for p in a1}) for ids in b.values()
        ),
        "current_sampling": {
            "coarse": len(coarse.points_scene_m),
            "fine": len(fine.points_scene_m),
            "common": int(common_old),
        },
        "current_split_changes": sum(
            old_split_small[key] != old_split_large[key] for key in old_split_small
        ),
        "a1_selected_hash": digest([p["id"] for p in a1]),
        "a2_selected_hash": digest([p["id"] for p in a2]),
    }
    return stats, tracks, base_points


def write_visuals(
    root: Path, output: Path, scenes: list[Scene], representatives: list[list[Track]]
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    figure = plt.figure(figsize=(12, 8))
    images = []
    review = []
    for index, (scene, tracks) in enumerate(zip(scenes, representatives, strict=True)):
        axis = figure.add_subplot(2, 2, index + 1, projection="3d")
        frame = tracks[0].center.scene_from_center.inverse()
        cameras = frame.apply(
            np.stack([c.camera_to_scene.matrix()[:3, 3] for c in scene.cameras])
        )
        axis.scatter(*cameras.T, s=2, c="grey", label="captured")
        for track in tracks[:4]:
            points = frame.apply(track.dense())
            axis.plot(*points.T, linewidth=0.8, label=track.spec.shape.value)
        for track in freeze_evaluation(scene):
            points = frame.apply(track.dense())
            axis.plot(*points.T, linestyle="--", linewidth=1, label=track.split)
        axis.set(
            title=scene.sid,
            xlabel="court-plane x (m)",
            ylabel="y (m)",
            zlabel="height (m)",
        )
        axis.legend(fontsize=6)
        owner = root / scene.sid / "datasets/court"
        manifest = read_court_manifest(owner)

        def target_count(record: dict[str, Any]) -> int:
            target_id = record["target_court"]["binding"]["court_instance_id"]
            court = next(
                c
                for c in record["projection"]["courts"]
                if c["court_instance_id"] == target_id
            )
            return sum(
                p["renderer_visible"] for c in court["classes"] for p in c["points"]
            )

        ranked = sorted(
            manifest["samples"], key=lambda p: (target_count(p), p["sample_id"])
        )
        store = open_court_store(owner)
        panels = []
        for name, record in [
            ("few target KPs", ranked[0]),
            ("many target KPs", ranked[-1]),
        ]:
            canvas = Image.fromarray(read_court_rgb(owner, record, store=store))
            draw = ImageDraw.Draw(canvas)
            for court in record["projection"]["courts"]:
                points = {
                    p["physical_index"]: p
                    for c in court["classes"]
                    for p in c["points"]
                }
                for a, b in COURT_SKELETON:
                    if (
                        a in points
                        and b in points
                        and points[a]["in_front"]
                        and points[b]["in_front"]
                    ):
                        draw.line(
                            [tuple(points[a]["uv"]), tuple(points[b]["uv"])],
                            fill=(50, 255, 100),
                            width=2,
                        )
                for point in points.values():
                    if point["in_frame"]:
                        x, y = point["uv"]
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 80, 30))
            canvas = canvas.resize((480, 270))
            ImageDraw.Draw(canvas).rectangle((0, 0, 480, 28), fill="black")
            ImageDraw.Draw(canvas).text(
                (5, 5),
                f"{scene.sid} saved RGB: {name} ({target_count(record)}/14)",
                fill="white",
            )
            panels.append(canvas)
            review.append(
                {
                    "scene": scene.sid,
                    "kind": name,
                    "sample_id": record["sample_id"],
                    "visible_target": target_count(record),
                }
            )
        image_row = Image.new("RGB", (960, 270))
        image_row.paste(panels[0], (0, 0))
        image_row.paste(panels[1], (480, 0))
        images.append(image_row)
        # CPU animation deliberately separates proposed geometry from saved RGB.
        frames = output / scene.sid
        frames.mkdir()
        track = tracks[0]
        path = frame.apply(track.dense())
        low, high = path[:, :2].min(axis=0) - 2, path[:, :2].max(axis=0) + 2

        def xy(point: Any, low: Any = low, high: Any = high) -> tuple[int, int]:
            q = (point[:2] - low) / (high - low)
            return int(40 + q[0] * 400), int(40 + (1 - q[1]) * 210)

        for tick, point in enumerate(
            plan(track, 2)[:: max(1, track.n0 * 4 // 32)][:32]
        ):
            canvas = Image.new("RGB", (960, 570), "#17212b")
            canvas.paste(image_row, (0, 300))
            draw = ImageDraw.Draw(canvas)
            draw.line([xy(p) for p in path], fill="#78bcff", width=2)
            position = frame.apply(np.array(point["pose"])[:3, 3][None])[0]
            target = frame.apply(np.array(point["look_at"])[None])[0]
            draw.line([xy(position), xy(target)], fill="yellow", width=2)
            x, y = xy(position)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
            draw.text(
                (10, 10),
                f"{scene.sid} PROPOSED geometry; no new 3DGS render",
                fill="white",
            )
            draw.text((500, 10), f"projected teacher: {point['reason']}", fill="white")
            if point["projection"] is not None:
                for court in point["projection"]["courts"]:
                    for cls in court["classes"]:
                        p = cls["points"][0]
                        if p["in_frame"]:
                            x, y = p["uv"]
                            x, y = (
                                480 + x / scene.cameras[0].width * 480,
                                y / scene.cameras[0].height * 270,
                            )
                            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#80ff80")
            canvas.save(frames / f"{tick:03d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-framerate",
                "8",
                "-i",
                str(frames / "%03d.png"),
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(output / f"{scene.sid}-review.mp4"),
            ],
            check=True,
        )
    figure.tight_layout()
    figure.savefig(output / "geometry.png", dpi=110)
    plt.close(figure)
    montage = Image.new("RGB", (960, 1080))
    for index, row in enumerate(images):
        montage.paste(row, (0, index * 270))
    montage.resize((640, 720)).save(output / "saved-overlays.png")
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visuals", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.scenes_root.resolve()):
        raise ValueError("output must not be inside the read-only input scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    scenes = [load_scene(args.scenes_root, sid) for sid in ["B00", "B01", "B02", "B03"]]
    results, tracks = [], []
    for scene in scenes:
        result, scene_tracks, _ = experiment(scene)
        results.append(result)
        tracks.append(scene_tracks)
        print(
            scene.sid,
            "common",
            result["common_geometry_equal"],
            "selected union",
            result["selected_union_count"],
            flush=True,
        )
    review = (
        write_visuals(args.scenes_root, args.output, scenes, tracks)
        if args.visuals
        else []
    )
    for scene in scenes:
        for name, expected in scene.hashes.items():
            if (
                hashlib.sha256(
                    (args.scenes_root / scene.sid / name).read_bytes()
                ).hexdigest()
                != expected
            ):
                raise ValueError(
                    f"read-only input changed during probe: {scene.sid}/{name}"
                )
    evidence = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "prototype_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_unchanged": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "probe": {
            "seed": SEED,
            "eval_seed": EVAL_SEED,
            "base_groups": BASE_GROUPS,
            "base_target": BASE_TARGET,
            "max_density_level": 4,
            "step_m": STEP_M,
            "guard_m": GUARD_M,
            "synthetic_acceptance_rate": 0.8,
        },
        "scenes": results,
        "saved_overlay_examples": review,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
