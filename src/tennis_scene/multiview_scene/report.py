"""Read small scene fields to report completion on the original frame axis."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_scene.inputs import (
    INPUT_NODE,
    RECIPE,
    FrozenInputs,
    read_payload,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.components.player_reconstruction import PlayerSkeleton
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.geometry.triangulation import PinholeCamera

SCENE_FIELDS = (
    "player_track_ids",
    "player_valid",
    "player_smpl_valid",
    "player_kp_3d_vis",
    "ball_3d_valid",
    "human_kp_2d",
    "human_kp_vis",
)


def scene_fields(store: StoreReader, names: tuple[str, ...]) -> dict[str, Any]:
    desc = store.descriptor("scene_assembly")
    ref = store.references["scene_assembly"]
    if store.document["exports"]["scene"]["inputs"] != {"scene_assembly": ref}:
        raise ValueError("Scene export and current assembly disagree")
    directory = (store.root / ref["path"]).parent
    return {
        name: unpack_value(desc["payload"][name], directory, desc["arrays"])
        for name in names
    }


def legacy_sides(legacy: StoreReader) -> dict[int, int]:
    if "player_association" not in legacy.references:
        return {}
    descriptor = legacy.descriptor("player_association")
    association = descriptor["payload"]["diagnostics"]["association"]
    sides = {r["identity"]: r["side"] for r in association["identities"]}
    return {r["player_id"]: sides[r["identity"]] for r in association["players"]}


def unused_reprojection(
    skeleton: PlayerSkeleton,
    cameras: tuple[PinholeCamera, ...],
    points_px: np.ndarray,
    visible: np.ndarray,
) -> dict[str, int | float | None]:
    residuals = []
    for view, camera in enumerate(cameras):
        projected, front = camera.project(skeleton.positions)
        usable = skeleton.valid & front & visible[:, view] & ~skeleton.inliers[:, view]
        residuals.append(
            np.linalg.norm(projected[usable] - points_px[:, view][usable], axis=-1)
        )
    values = np.concatenate(residuals)
    return {
        "points": int(values.size),
        "median_px": float(np.median(values)) if len(values) else None,
        "p95_px": float(np.quantile(values, 0.95)) if len(values) else None,
    }


def completion(
    clip_id: str,
    frames: int,
    ids: tuple[int, ...],
    sides: dict[int, int],
    players_per_side: int,
    body: GVHMROutput | None,
    fields: dict[str, Any],
    *,
    reason: str = "",
) -> dict[str, Any]:
    by_side = {side: [i for i in ids if sides.get(i) == side] for side in (-1, 1)}
    present: NDArray[np.bool_] = np.zeros((len(ids), frames), dtype=bool)
    if body is not None:
        for person in body.bodies:
            if person.person_id not in ids:
                raise ValueError("Recovered body references an unknown selected ID")
            for segment in person.segments:
                present[ids.index(person.person_id), segment.source_frames] = True
    shape = (len(ids), frames)
    mesh: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    root: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    joints: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    if fields:
        if fields["player_track_ids"].tolist() != list(ids):
            raise ValueError("Scene player IDs differ from the completion table")
        mesh, root = fields["player_smpl_valid"], fields["player_valid"]
        joints = fields["player_kp_3d_vis"].any(axis=-1)
    all_sides = all(len(by_side[side]) == players_per_side for side in by_side)
    ball = int(fields["ball_3d_valid"].sum()) if fields else 0
    reasons = [reason] if reason else []
    if not all_sides:
        reasons.append("selection_shortfall")
    if not fields:
        reasons.append("scene_missing")
    elif not ball:
        reasons.append("ball_3d_missing")
    if ids and not present.any(axis=1).all():
        reasons.append("gvhmr_missing_for_selected_id")
    if ids and not mesh.any(axis=1).all():
        reasons.append("placed_body_missing_for_selected_id")
    full_bodies = int(present.all(axis=0).sum()) if all_sides else 0
    full_placement = int(mesh.all(axis=0).sum()) if all_sides else 0
    return {
        "clip_id": clip_id,
        "source_frames": frames,
        "players_per_side": players_per_side,
        "selected_negative": len(by_side[-1]),
        "selected_positive": len(by_side[1]),
        "selected_ids_negative": json.dumps(by_side[-1]),
        "selected_ids_positive": json.dumps(by_side[1]),
        "gvhmr_frames_by_id": json.dumps(
            dict(zip(ids, present.sum(axis=1).tolist(), strict=True))
        ),
        "valid_placement_frames_by_id": json.dumps(
            dict(zip(ids, mesh.sum(axis=1).tolist(), strict=True))
        ),
        "valid_root_frames_by_id": json.dumps(
            dict(zip(ids, root.sum(axis=1).tolist(), strict=True))
        ),
        "all_gvhmr_frames": full_bodies,
        "all_gvhmr_ratio": full_bodies / frames,
        "all_placed_frames": full_placement,
        "all_placed_ratio": full_placement / frames,
        "joint_player_frames": int(joints.sum()),
        "joint_coverage_ratio": int(joints.sum()) / (frames * 2 * players_per_side),
        "ball_3d_frames": ball,
        "ball_3d_ratio": ball / frames,
        "complete": not reasons,
        "reason": " | ".join(reasons),
    }


def report_clip(
    store: StoreReader, *, players_per_side: int, new: bool, reason: str = ""
) -> dict[str, Any]:
    index = store.document
    body = None
    skeleton: PlayerSkeleton | None = None
    cameras: tuple[PinholeCamera, ...] = ()
    source_frames = store.frame_count
    fields: dict[str, Any] = {}
    if index["exports"].get("scene"):
        fields = scene_fields(store, SCENE_FIELDS)
    ids = tuple(map(int, fields["player_track_ids"])) if fields else ()
    sides: dict[int, int] = {}
    if new:
        if INPUT_NODE in store.references:
            frozen = read_payload(store, INPUT_NODE, INPUT_NODE, 1, FrozenInputs)
            if frozen.pins["recipe"] != RECIPE:
                raise ValueError("Output store has the wrong reconstruction recipe")
            ids = tuple(map(int, frozen.people.grouped.identities))
            sides = dict(zip(ids, frozen.people.sides, strict=True))
            skeleton = frozen.people.skeleton
            if frozen.alignment.geometry is not None:
                cameras = frozen.alignment.geometry.cameras
    else:
        sides = legacy_sides(store) if ids else {}
        if ids and fields:
            from src.tennis_scene.pipeline.components.camera_alignment import (
                CameraAlignmentOutput,
            )

            deps = store.descriptor("scene_assembly")["dependencies"]
            node_by_id = {
                ref["artifact_id"]: node for node, ref in store.references.items()
            }
            skeleton_node = node_by_id[deps["skeleton"]["artifact_id"]]
            alignment_node = node_by_id[deps["alignment"]["artifact_id"]]
            skeleton = read_payload(
                store, skeleton_node, "player_skeletons", 1, PlayerTriangulationOutput
            ).skeleton
            alignment = read_payload(
                store, alignment_node, "aligned_cameras", 1, CameraAlignmentOutput
            )
            if alignment.geometry is not None:
                cameras = alignment.geometry.cameras
    if ids and "gvhmr" in store.references:
        body = read_payload(store, "gvhmr", "body_parameters", 2, GVHMROutput)
    report = completion(
        store.source["clip_id"],
        source_frames,
        ids,
        sides,
        players_per_side,
        body,
        fields,
        reason=reason,
    )
    if skeleton is not None and cameras and fields:
        indices = [store.camera_ids.index(camera.camera_id) for camera in cameras]
        px = fields["human_kp_2d"][:, indices] * np.asarray(store.size, np.float32)
        reprojection = unused_reprojection(
            skeleton, cameras, px, fields["human_kp_vis"][:, indices] > 0
        )
    else:
        reprojection = {"points": 0, "median_px": None, "p95_px": None}
    report.update({f"unused_view_{key}": value for key, value in reprojection.items()})
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(buffer.getvalue())
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
