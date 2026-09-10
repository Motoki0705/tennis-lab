"""Explicit import and verification of the editor's immutable evidence frame."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentResult,
    GroundPlaneFrame,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    validate_line_heatmaps,
)
from src.synthetic_data_generation.alignment.manual.models import (
    CourtPlacement,
    LayoutEdit,
)
from src.synthetic_data_generation.alignment.validation import load_alignment_result

SOURCE_SCHEMA = "manual_alignment_source_v1"


def file_digest(path: Path) -> str:
    """Hash ordinary files without following an untrusted symlink."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected an ordinary source file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def owner_digest(root: Path) -> str:
    """Bind both inventory and bytes, including diagnostic rasters."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source symlinks are unsupported: {path}")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(file_digest(path).encode())
    return digest.hexdigest()


def reconstruction_identity(scene_root: Path) -> dict[str, str]:
    """Bind manual geometry to the public reconstruction coordinate authority."""
    export = scene_root / "reconstruction" / "export"
    return {
        name: file_digest(export / name)
        for name in ("scene.json", "cameras.json", "points_scene.npy")
    }


@dataclass(frozen=True)
class ManualSource:
    """Initial geometry is reference only; human placement is the final authority."""

    initial: AlignmentResult
    plane: GroundPlaneFrame
    provenance: dict[str, Any]
    reconstruction: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_SCHEMA,
            "initial_alignment": self.initial.to_dict(),
            "ground_plane": self.plane.to_dict(),
            "provenance": self.provenance,
            "reconstruction": self.reconstruction,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManualSource:
        if (
            set(raw)
            != {
                "schema",
                "initial_alignment",
                "ground_plane",
                "provenance",
                "reconstruction",
            }
            or raw["schema"] != SOURCE_SCHEMA
        ):
            raise ValueError("Invalid manual source schema.")
        initial = AlignmentResult.from_dict(raw["initial_alignment"])
        if any(c.human_confirmed for c in initial.candidates):
            raise ValueError(
                "Manual source must retain the original automatic reference."
            )
        if set(raw["reconstruction"]) != {
            "scene.json",
            "cameras.json",
            "points_scene.npy",
        }:
            raise ValueError("Invalid reconstruction identity.")
        return cls(
            initial,
            GroundPlaneFrame.from_dict(raw["ground_plane"]),
            raw["provenance"],
            raw["reconstruction"],
        )

    def initial_edit(self) -> LayoutEdit:
        placements = []
        u = np.asarray(self.plane.basis_u_metric_scene)
        v = np.asarray(self.plane.basis_v_metric_scene)
        for court in self.initial.layout.courts:
            matrix = court.scene_from_court.matrix()
            center = self.plane.to_uv(matrix[None, :3, 3])[0]
            angle = np.degrees(np.arctan2(matrix[:3, 0] @ v, matrix[:3, 0] @ u))
            placements.append(
                CourtPlacement(
                    court_id=court.court_instance_id,
                    u=float(center[0]),
                    v=float(center[1]),
                    angle_degrees=float(angle),
                )
            )
        return LayoutEdit(
            scale=1.0,
            courts=placements,
            primary_court_id=self.initial.layout.primary_court_instance_id,
        )


def import_source(
    scene_root: Path, *, recover_ground_frame: bool
) -> tuple[ManualSource, AlignmentLineHeatmaps]:
    """Import an automatic owner; older frames require explicit verified recovery."""
    root = scene_root / "alignment"
    heatmaps = validate_line_heatmaps(root / "line-heatmaps")
    initial = load_alignment_result(root / "alignment.json")
    with np.load(root / "ground-line-map.npz", allow_pickle=False) as archive:
        provenance: dict[str, Any] = {
            "alignment_sha256": owner_digest(root),
            "archive_schema": str(archive["schema"].item()),
        }
        if not np.allclose(
            archive["nht_scene_from_metric_scene"],
            initial.metric_adapter.nht_matrix(),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("Source archive and alignment metric adapters disagree.")
        ids = list(archive["fit_camera_ids"]) + list(archive["holdout_camera_ids"])
        expected_ids = list(
            initial.partitions.fit_camera_ids + initial.partitions.holdout_camera_ids
        )
        if ids != expected_ids:
            raise ValueError(
                "Source archive camera partitions disagree with alignment."
            )
        line_points = archive["line_points_nht_scene"]
        line_indices = archive["line_camera_index"]
        uv, xyz = [], []
        for view in heatmaps.views:
            if view.camera_id not in ids:
                if view.included_in_aggregate:
                    raise ValueError(
                        "An unretained camera cannot contribute fit evidence."
                    )
                continue
            points = line_points[line_indices == ids.index(view.camera_id)]
            if len(points) != len(view.points_uv):
                raise ValueError("Paired UV/3D observation counts differ.")
            uv.append(view.points_uv)
            xyz.append(initial.metric_adapter.metric_from_nht_points(points))
        points_uv = np.concatenate(uv)
        points_metric = np.concatenate(xyz)
        if "ground_plane_frame_json" in archive:
            plane = GroundPlaneFrame.from_dict(
                json.loads(str(archive["ground_plane_frame_json"].item()))
            )
            if (
                not np.allclose(
                    plane.to_uv(points_metric), points_uv, atol=1e-7, rtol=0
                )
                or np.max(np.abs(plane.signed_distances(points_metric))) > 1e-7
            ):
                raise ValueError(
                    "Persisted plane disagrees with paired UV/3D evidence."
                )
            provenance["frame_method"] = "persisted"
        else:
            if not recover_ground_frame:
                raise ValueError(
                    "Ground frame is absent. Use --recover-ground-frame to explicitly verify and recover it from paired UV/3D observations."
                )
            design = np.column_stack((points_uv, np.ones(len(points_uv))))
            coefficients, _, rank, _ = np.linalg.lstsq(
                design, points_metric, rcond=None
            )
            residual = float(np.max(np.abs(design @ coefficients - points_metric)))
            if rank != 3 or residual > 1e-7:
                raise ValueError(
                    f"Ground frame recovery failed: rank={rank}, residual={residual}."
                )
            plane = GroundPlaneFrame(
                origin_metric_scene=tuple(coefficients[2]),
                basis_u_metric_scene=tuple(coefficients[0]),
                basis_v_metric_scene=tuple(coefficients[1]),
                normal_metric_scene=tuple(np.cross(coefficients[0], coefficients[1])),
                bounds_uv_metres=heatmaps.bounds_uv,
            )
            provenance.update(
                frame_method="verified_paired_uv_3d", maximum_residual_metres=residual
            )
    if plane.bounds_uv_metres != heatmaps.bounds_uv:
        raise ValueError("Ground plane and heatmap bounds disagree.")
    source = ManualSource(
        initial, plane, provenance, reconstruction_identity(scene_root)
    )
    validate_source_heatmaps(source, heatmaps)
    return source, heatmaps


def validate_source_heatmaps(
    source: ManualSource, heatmaps: AlignmentLineHeatmaps
) -> None:
    """Require the original frame, camera inventory, and split ownership."""
    partitions = source.initial.partitions
    if source.plane.bounds_uv_metres != heatmaps.bounds_uv:
        raise ValueError("Source plane and heatmap bounds disagree.")
    if not set(partitions.fit_camera_ids + partitions.holdout_camera_ids).issubset(
        heatmaps.camera_ids
    ):
        raise ValueError("Source camera inventory differs from heatmaps.")
    if set(heatmaps.aggregate_camera_ids) != set(partitions.fit_camera_ids):
        raise ValueError("Source fit ownership differs from heatmaps.")
