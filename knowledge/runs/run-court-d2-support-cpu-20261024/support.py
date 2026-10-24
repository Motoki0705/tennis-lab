"""CPU-only court framing diagnostics; real geometry is restricted to val IDs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _look_at_opencv,
)
from src.synthetic_data_generation.dataset.court.scale.core import GeometryRecipe
from src.synthetic_data_generation.dataset.court.scale.runtime import load_scene_inputs
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.data.image_record_store import file_sha256
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "run-court-scale-pilot-l4-20261023"
ASSETS = HERE.parent / "run-court-assets-colab-l4-20261022/assets.json"
SCENES = ("B00", "B01", "B02", "B03")
GROUPS = ("full", "near_full", "partial")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recipe() -> GeometryRecipe:
    source = Path(
        "src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json"
    )
    return GeometryRecipe.from_dict(json.loads(source.read_text())["geometry"])


def load_inputs(scenes_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Check source bytes against the adopted bundle without copying or modifying them."""
    manifest = json.loads(ASSETS.read_text())
    assert (
        file_sha256(ASSETS)
        == "a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b"
    )
    checked = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        assert relative.parts[0] == "scenes"
        source = scenes_root / relative.relative_to("scenes")
        assert source.stat().st_size == entry["size_bytes"]
        assert file_sha256(source) == entry["sha256"], source
        checked.append(entry["path"])
    inputs = {
        scene: load_scene_inputs(scenes_root / scene, replay=False) for scene in SCENES
    }
    return inputs, {"manifest_sha256": file_sha256(ASSETS), "verified_files": checked}


def framing(count: int) -> str:
    if not 0 <= count <= 14:
        raise ValueError("Expected KP14 count")
    return "full" if count == 14 else "near_full" if count >= 12 else "partial"


def target_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    target = record["target_court"]["binding"]["court_instance_id"]
    courts = [
        c for c in record["projection"]["courts"] if c["court_instance_id"] == target
    ]
    assert len(courts) == 1
    classes = sorted(courts[0]["classes"], key=lambda c: c["class_id"])
    assert [c["class_id"] for c in classes] == list(range(14))
    return [c["points"][0] for c in classes]


def count_points(points: list[dict[str, Any]], width: int, height: int) -> int:
    return int(
        sum(
            p["in_front"] and 0 <= p["uv"][0] < width and 0 <= p["uv"][1] < height
            for p in points
        )
    )


def physical_projection(
    record: dict[str, Any], layout: Any, camera: SceneCamera
) -> list[dict[str, Any]]:
    """Project physical KP14 for diagnostics, even when semantic near/far is undefined.

    This never supplies missing production labels. Geometry-rejected candidates
    retain their rejection; their unordered framing can still be measured.
    """
    court = next(
        c
        for c in layout.courts
        if c.court_instance_id == record["target_court"]["binding"]["court_instance_id"]
    )
    world = court.scene_from_court.apply(
        court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(np.float64)
    )
    xyz = camera.camera_to_scene.inverse().apply(world)
    homogeneous = xyz @ np.array(camera.intrinsics).reshape(3, 3).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:]
    return [
        {"uv": pixel, "in_front": bool(point[2] > 0), "scene_xyz_m": pos}
        for pixel, point, pos in zip(uv, xyz, world, strict=True)
    ]


def region(x: float, y: float) -> str:
    end, side = abs(y) > HALF_LENGTH, abs(x) > HALF_DOUBLES_WIDTH
    return (
        "corner_exterior"
        if end and side
        else "baseline_exterior"
        if end
        else "sideline_exterior"
        if side
        else "inside_court"
    )


def describe(
    record: dict[str, Any], group: dict[str, Any], scene: str, layout: Any, hull: Any
) -> dict[str, Any]:
    camera = SceneCamera.from_dict(record["camera"])
    position = camera.camera_to_scene.matrix()[:3, 3]
    court = next(
        c
        for c in layout.courts
        if c.court_instance_id == record["target_court"]["binding"]["court_instance_id"]
    )
    local = court.court_from_scene.apply(position[None])[0]
    capture_local = hull[0].inverse().apply(position[None])[0]
    clearance = float(
        np.min(
            -(hull[1].equations[:, :2] @ capture_local[:2] + hull[1].equations[:, 2])
        )
    )
    points = physical_projection(record, layout, camera)
    count = count_points(points, camera.width, camera.height)
    if record["projection"] is not None:
        assert count == count_points(target_points(record), camera.width, camera.height)
    else:
        assert any(
            "ambiguous_camera_relative_near_far" in reason
            for reason in record["reasons"]
        )
    uv = np.array([p["uv"] for p in points])
    front = np.array([p["in_front"] for p in points])
    horizontal = ((uv[:, 0] < 0) | (uv[:, 0] >= camera.width)) & front
    vertical = ((uv[:, 1] < 0) | (uv[:, 1] >= camera.height)) & front
    xyz = camera.camera_to_scene.inverse().apply(
        np.array([p["scene_xyz_m"] for p in points])
    )
    required = None
    if front.all():
        xy = xyz[:, :2] / xyz[:, 2:]
        cx, cy = np.array(camera.intrinsics).reshape(3, 3)[:2, 2]
        # Infimum HFOV at the half-open pixel border, retaining the exact pose.
        inverse_focal_limit = max(
            (xy[:, 0] / (camera.width - cx)).max(),
            (-xy[:, 0] / cx).max(),
            (xy[:, 1] / (camera.height - cy)).max(),
            (-xy[:, 1] / cy).max(),
        )
        required = math.degrees(2 * math.atan(camera.width * inverse_focal_limit / 2))
    return {
        "scene": scene,
        "sample_id": record["sample_id"],
        "pair_id": record["scale"]["comparison_pair_id"],
        "ordinal": group["ordinal"],
        "geometry_pass": not record["reasons"],
        "geometry_reasons": record["reasons"],
        "semantic_projection_available": record["projection"] is not None,
        "framing": framing(count),
        "in_frame_kp": count,
        "height_m": float(local[2]),
        "distance_xy_m": float(np.linalg.norm(local[:2])),
        "x_m": float(local[0]),
        "y_m": float(local[1]),
        "region": region(local[0], local[1]),
        "hfov_deg": group["view"]["hfov_degrees"],
        "look_at_height_m": group["view"]["look_at_height_m"],
        "hull_clearance_m": clearance,
        "required_hfov_deg": required,
        "all_kp_in_front": bool(front.all()),
        "horizontal_clipping": bool(horizontal.any()),
        "vertical_clipping": bool(vertical.any()),
        "behind_camera": bool((~front).any()),
        "miss_pattern": "+".join(
            name
            for name, yes in (
                ("behind", (~front).any()),
                ("horizontal", horizontal.any()),
                ("vertical", vertical.any()),
            )
            if yes
        )
        or "none",
    }


def hull_for(data: dict[str, Any]) -> tuple[Any, Any]:
    from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
        derive_orbit_centers,
    )

    centers = derive_orbit_centers(
        data["cameras"],
        data["layout"],
        use_court_centroid=True,
        use_captured_hull_centroid=True,
    )
    reference = centers[0].scene_from_center
    local = reference.inverse().apply(
        np.stack([c.camera_to_scene.matrix()[:3, 3] for c in data["cameras"]])
    )
    return reference, ConvexHull(local[:, :2])


def stats(values: list[float]) -> dict[str, Any]:
    return (
        {
            "n": len(values),
            "min_p10_median_p90_max": np.percentile(
                values, [0, 10, 50, 90, 100]
            ).tolist(),
        }
        if values
        else {"n": 0}
    )


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "framing": {g: sum(r["framing"] == g for r in rows) for g in GROUPS},
        "in_frame_kp": dict(sorted(Counter(r["in_frame_kp"] for r in rows).items())),
        "miss_pattern": dict(Counter(r["miss_pattern"] for r in rows)),
        "outside_captured_hull": sum(r["hull_clearance_m"] < 0 for r in rows),
        **{
            field: stats([float(r[field]) for r in rows if r[field] is not None])
            for field in (
                "height_m",
                "distance_xy_m",
                "hull_clearance_m",
                "required_hfov_deg",
            )
        },
    }


def partitions(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[field])].append(row)
    return {key: summary(selected) for key, selected in sorted(values.items())}


def counterfactual(
    record: dict[str, Any],
    data: dict[str, Any],
    *,
    hfov: float | None = None,
    target_height: float | None = None,
) -> str:
    camera = SceneCamera.from_dict(record["camera"])
    if hfov is not None:
        k = np.array(camera.intrinsics).reshape(3, 3).copy()
        k[0, 0] = k[1, 1] = camera.width / (2 * math.tan(math.radians(hfov) / 2))
        camera = replace(camera, intrinsics=tuple(k.ravel()))
    if target_height is not None:
        court = next(
            c
            for c in data["layout"].courts
            if c.court_instance_id
            == record["target_court"]["binding"]["court_instance_id"]
        )
        look_at = np.array(record["scale"]["look_at_scene_m"])
        local = court.court_from_scene.apply(look_at[None])[0]
        local[2] = target_height
        pose = _look_at_opencv(
            camera.camera_to_scene.matrix()[:3, 3],
            court.scene_from_court.apply(local[None])[0],
            vertical_scene=court.scene_from_court.matrix()[:3, 2],
        )
        camera = replace(camera, camera_to_scene=RigidTransform.from_matrix(pose))
    return framing(
        count_points(
            physical_projection(record, data["layout"], camera),
            camera.width,
            camera.height,
        )
    )


def row_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
