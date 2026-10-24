"""Pin paired novel-view probes; intentionally stratified, not a random survey."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from src.synthetic_data_generation.dataset.court.components.labels import (
    AmbiguousCameraRelativeNearFarError,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderCamera,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera

# The previous CPU bundle is an explicit, unchanged dependency.
prior = importlib.import_module(
    "knowledge.runs.run-court-nested-plan-cpu-20261022.prototype"
)

CONDITIONS = {
    "recommended": {"expansion_percent": 0.0, "jitter_m": 1.0},
    "hull5": {"expansion_percent": 5.0, "jitter_m": 1.0},
    "hfov45_90": {"expansion_percent": 0.0, "jitter_m": 1.0},
    "jitter0": {"expansion_percent": 0.0, "jitter_m": 0.0},
}
HFOVS = [75.0, 92.5, 110.0]
NARROW_HFOVS = [45.0, 67.5, 90.0]


def dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def camera_from_record(record: dict[str, Any]) -> SceneCamera:
    return SceneCamera(
        record["id"],
        0,
        record["width"],
        record["height"],
        tuple(record["K"]),
        RigidTransform.from_matrix(record["pose"]),
        "probe-only",
    )


def paired_track(scene: Any, ordinal: int, expansion: float) -> Any:
    track = prior.make_track(scene, ordinal)
    centers = prior.derive_orbit_centers(
        scene.cameras,
        scene.alignment.layout,
        use_court_centroid=True,
        use_captured_hull_centroid=True,
    )
    unbounded = replace(track.spec, base_radius_m=track.center.base_radius_m)
    spec = prior.bound_trajectory_candidates(
        [unbounded],
        centers=centers,
        cameras=scene.cameras,
        margin_m=0.5,
        expansion_percent=expansion,
    )[0]
    theta = np.linspace(0, 2 * math.pi, prior.DENSE_COUNT + 1)
    local = prior._points_local(spec, theta)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
    return replace(
        track,
        spec=spec,
        dense_theta=theta,
        cumulative=cumulative,
        length=float(cumulative[-1]),
    )


def make_record(
    track: Any,
    condition: str,
    fraction: Fraction,
    hfov_layer: int,
    kind: str,
    index: int,
    ordinal: int,
) -> dict[str, Any]:
    policy = CONDITIONS[condition]
    fraction_key = f"{fraction.numerator}/{fraction.denominator}"
    point_id = prior.digest(
        ["novel-view-probe-v1", track.scene.sid, ordinal, fraction_key, hfov_layer]
    )
    theta = np.interp(
        float(fraction) * track.length, track.cumulative, track.dense_theta
    )
    position = track.center.scene_from_center.apply(
        prior._points_local(track.spec, np.array([theta]))
    )[0]
    target = prior.resolve_target_court(
        policy=prior.target_court_policy_for_trajectory(track.spec),
        camera_center_scene_m=position,
        layout=track.scene.alignment.layout,
        selection_seed=prior.SEED,
    )
    target_id = target.binding.court_instance_id
    court = track.scene.alignment.layout.court(target_id)
    random = prior.rng_for("novel-view-look-at-v1", prior.SEED, point_id)
    radius = math.sqrt(float(random.random())) * policy["jitter_m"]
    angle = 2 * math.pi * float(random.random())
    look_at = court.scene_from_court.apply(
        np.array(
            [[radius * math.cos(angle), radius * math.sin(angle), track.look_height]]
        )
    )[0]
    pose = prior._look_at_opencv(
        position, look_at, vertical_scene=track.center.scene_from_center.matrix()[:3, 2]
    )
    hfov = (NARROW_HFOVS if condition == "hfov45_90" else HFOVS)[hfov_layer]
    width, height = track.scene.cameras[0].width, track.scene.cameras[0].height
    k = prior._intrinsics_from_hfov(width=width, height=height, hfov_degrees=hfov)
    camera = SceneCamera(
        "v-" + point_id,
        0,
        width,
        height,
        k,
        RigidTransform.from_matrix(pose),
        "probe-only",
    )
    reference = track.center.scene_from_center.inverse()
    captured = reference.apply(
        np.stack([c.camera_to_scene.matrix()[:3, 3] for c in track.scene.cameras])
    )
    local_position = reference.apply(position[None])[0]
    hull = ConvexHull(captured[:, :2])
    clearance = float(
        -(hull.equations[:, :2] @ local_position[:2] + hull.equations[:, 2]).max()
    )
    points = []
    ambiguous = False
    coverage = "undefined"
    visible_all = visible_target = 0
    try:
        projection = project_court_semantics_v3(camera, track.scene.alignment.layout)
        visible_all = sum(c.in_frame_point_count for c in projection.courts)
        projected = next(
            c for c in projection.courts if c.court_instance_id == target_id
        )
        points = [p.to_dict() for cls in projected.classes for p in cls.points]
        coverage = projected.coverage_mode
        visible_target = projected.in_frame_point_count
    except AmbiguousCameraRelativeNearFarError:
        ambiguous = True
    record = {
        "id": camera.camera_id,
        "pair_id": point_id,
        "scene": track.scene.sid,
        "condition": condition,
        "kind": kind,
        "index": index,
        "ordinal": ordinal,
        "fraction": fraction_key,
        "hfov_layer": hfov_layer,
        "hfov": hfov,
        "shape": track.spec.shape.value,
        "center_kind": track.spec.center_kind.value,
        "height_m": float(local_position[2]),
        "look_height_m": track.look_height,
        "hull_clearance_m": clearance,
        "target": target_id,
        "look_at": look_at.tolist(),
        "position": position.tolist(),
        "K": list(k),
        "pose": pose.tolist(),
        "width": width,
        "height": height,
        "geometry_target_kp": visible_target,
        "geometry_all_kp": visible_all,
        "coverage": coverage,
        "ambiguous": ambiguous,
        "existing_pre_gate": not ambiguous and visible_all >= 4,
        "points": points,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.scenes_root.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError(
            "Output must be a new directory outside the read-only input root"
        )
    output.mkdir(parents=True)
    inventory = {}
    for sid in ["B00", "B01", "B02", "B03"]:
        scene = prior.load_scene(root, sid)
        export = prior.validate_standard_scene_export(
            root / sid / "reconstruction/export/scene.json"
        )
        hashes = dict(scene.hashes)
        for path in [export.checkpoint_path, export.runtime_config_path]:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(block)
            hashes[str(path.relative_to(root / sid))] = h.hexdigest()
        inventory[sid] = {"hashes": hashes, "compatibility": scene.compatibility}
        # Sparse probes use min/max target-KP visibility within each FOV layer.
        # Six paths span shapes/centres/heights. Selection uses geometry only.
        sparse = []
        for ordinal in [0, 1, 2, 7, 12, 15]:
            track = paired_track(scene, ordinal, 0.0)
            for layer in range(3):
                pool = [
                    make_record(
                        track,
                        "recommended",
                        Fraction(i, 24),
                        layer,
                        "sparse",
                        0,
                        ordinal,
                    )
                    for i in range(24)
                ]
                ranked = sorted(
                    pool,
                    key=lambda r: (
                        r["geometry_target_kp"],
                        r["hull_clearance_m"],
                        r["fraction"],
                    ),
                )
                for r in [ranked[0], ranked[-1]]:
                    sparse.append((ordinal, Fraction(r["fraction"]), layer))
        for condition, policy in CONDITIONS.items():
            records = []
            # One complete continuous complex circle, fixed 92.5-degree layer.
            track = paired_track(scene, 0, policy["expansion_percent"])
            for i in range(96):
                records.append(
                    make_record(
                        track, condition, Fraction(i, 96), 1, "continuous", i, 0
                    )
                )
            for i, (ordinal, fraction, layer) in enumerate(sparse):
                track = paired_track(scene, ordinal, policy["expansion_percent"])
                records.append(
                    make_record(track, condition, fraction, layer, "sparse", i, ordinal)
                )
            # Reuse exact cameras appearing in both sparse and continuous inventories.
            unique = {r["id"]: r for r in records}
            request = NHTRenderRequest(
                tuple(
                    NHTRenderCamera.from_scene_camera(
                        replace(
                            camera_from_record(r),
                            camera_to_scene=scene.alignment.metric_adapter.nht_from_metric_camera(
                                camera_from_record(r).camera_to_scene
                            ),
                        )
                    )
                    for r in unique.values()
                )
            )
            request.write(output / sid / condition / "request.json")
            dump(output / sid / condition / "plan.json", records)
            if sid == "B00" and condition == "recommended":
                NHTRenderRequest(request.cameras[:1]).write(output / "smoke-1.json")
                NHTRenderRequest(request.cameras[1:5]).write(output / "smoke-4.json")
            print(
                sid,
                condition,
                len(records),
                "records",
                len(unique),
                "unique cameras",
                flush=True,
            )
    dump(output / "inputs.json", inventory)
    dump(
        output / "policy.json",
        {
            "conditions": CONDITIONS,
            "hfov_layers": HFOVS,
            "alternative_hfov_layers": NARROW_HFOVS,
            "continuous_points": 96,
            "sparse_points": 36,
            "seed": prior.SEED,
            "selection": "six paths x three FOVs x min/max geometry target KP; rates are diagnostic, not population estimates",
        },
    )


if __name__ == "__main__":
    main()
