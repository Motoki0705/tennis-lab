"""Audit temporal ground-height disagreement in the saved B00 sparse SfM model.

COLMAP's documented binary tracks identify observation frames, not acquisition
times of the reconstructed coordinates. Distinct short tracks are compared in
shared ground cells; these measurements are not absolute drift ground truth.
Binary format: https://colmap.github.io/format.html
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from common import REPO, ROOT, sha256, write_json
from scipy.spatial import cKDTree

BUNDLE = ROOT / "evidence/sfm_drift"
METHOD = ROOT / "evidence/alignment_method"
ALIGNMENT = ROOT / "evidence/scene_sources/B00.json"
SCENE = REPO / "data/synthetic_data_generation/scenes/B00"


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated COLMAP model")
    return value


def read_tracks(model: Path) -> tuple[np.ndarray, np.ndarray, dict, list[int]]:
    """Read public COLMAP points/tracks without pycolmap or NHT internals."""
    frames = {}
    with (model / "images.bin").open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(count):
            image = struct.unpack("<idddddddi", read_exact(stream, 64))
            name = bytearray()
            while (char := read_exact(stream, 1)) != b"\0":
                name.extend(char)
            stem = Path(name.decode("utf-8")).stem
            if not stem.startswith("frame_") or not stem[6:].isdigit():
                raise ValueError(f"Unexpected B00 frame name: {stem}")
            if image[0] in frames:
                raise ValueError("Duplicate COLMAP image ID")
            frames[image[0]] = int(stem[6:])
            n = struct.unpack("<Q", read_exact(stream, 8))[0]
            read_exact(stream, 24 * n)  # x, y, point3D ID per observation
        if stream.read(1):
            raise ValueError("Trailing COLMAP image data")
    xyz, rgb, rows = [], [], []
    with (model / "points3D.bin").open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(count):
            point = struct.unpack("<QdddBBBd", read_exact(stream, 43))
            n = struct.unpack("<Q", read_exact(stream, 8))[0]
            if n < 2:
                raise ValueError("SfM point has fewer than two observations")
            track = np.frombuffer(read_exact(stream, 8 * n), dtype="<i4").reshape(n, 2)
            observed = [frames[int(i)] for i in track[:, 0]]
            xyz.append(point[1:4])
            rgb.append(point[4:7])
            rows.append((point[0], min(observed), max(observed), n, point[7]))
        if stream.read(1):
            raise ValueError("Trailing COLMAP point data")
    # IDs are not necessarily consecutive or in the export's point order.
    fields = {
        "point_id": np.array([r[0] for r in rows], dtype=np.uint64),
        "first_frame": np.array([r[1] for r in rows], dtype=np.int32),
        "last_frame": np.array([r[2] for r in rows], dtype=np.int32),
        "track_length": np.array([r[3] for r in rows], dtype=np.int32),
        "reprojection_error_px": np.array([r[4] for r in rows], dtype=np.float64),
    }
    if len(np.unique(fields["point_id"])) != count:
        raise ValueError("Duplicate COLMAP point ID")
    return np.array(xyz), np.array(rgb, dtype=np.uint8), fields, sorted(frames.values())


def match_export(
    xyz: np.ndarray, rgb: np.ndarray, exported: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, float]:
    """Require a bijection, bounded coordinate error, and exact exported RGB."""
    native = xyz @ transform[:3, :3].T + transform[:3, 3]
    distance, indices = cKDTree(exported[:, :3]).query(native)
    if (
        len(native) != len(exported)
        or len(np.unique(indices)) != len(exported)
        or distance.max() > 5e-6
        or not np.array_equal((rgb / 255).astype(np.float32), exported[indices, 3:])
    ):
        raise ValueError("COLMAP points do not map bijectively to the public export")
    return np.argsort(indices), float(distance.max())


def paired_heights(
    uv: np.ndarray,
    height: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    eligible: np.ndarray,
    edges: np.ndarray,
    cell_size: float,
    minimum: int,
) -> dict:
    """Use only cells supported in every period, never unpaired regional means."""
    cells = np.floor(uv / cell_size).astype(np.int64)
    per_period, point_counts, cell_counts = [], [], []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        # A track crossing a temporal boundary must not appear in either period.
        chosen = eligible & (first >= start) & (last < stop)
        unique, inverse, counts = np.unique(
            cells[chosen], axis=0, return_inverse=True, return_counts=True
        )
        values = height[chosen]
        medians = {
            tuple(cell): float(np.median(values[inverse == i]))
            for i, (cell, count) in enumerate(zip(unique, counts, strict=True))
            if count >= minimum
        }
        per_period.append(medians)
        point_counts.append(int(chosen.sum()))
        cell_counts.append(len(medians))
    shared = sorted(set.intersection(*(set(p) for p in per_period)))
    if not shared:
        raise ValueError("No ground cells shared by all temporal periods")
    medians = np.array([[p[cell] for cell in shared] for p in per_period])
    delta = medians - medians[0]
    return {
        "cell_size_m": cell_size,
        "qualified_point_counts": point_counts,
        "qualified_cell_counts": cell_counts,
        "shared_cell_count": len(shared),
        "shared_cell_indices": np.array(shared).tolist(),
        "cell_median_height_m": medians.tolist(),
        "cell_delta_from_first_m": delta.tolist(),
        "median_delta_m": np.median(delta, axis=1).tolist(),
        "iqr_delta_m": np.quantile(delta, [0.25, 0.75], axis=1).T.tolist(),
    }


def geometry(points: np.ndarray, method: dict, alignment: dict) -> dict:
    xyz = points[:, :3].astype(np.float64) / method["nht_scene_units_per_metre"]
    frame = method["ground_plane_frame"]
    centered = xyz - frame["origin_metric_scene"]
    basis = np.array([frame["basis_u_metric_scene"], frame["basis_v_metric_scene"]]).T
    inside = np.zeros(len(xyz), dtype=bool)
    outlines = []
    # Regulation doubles rectangle in each saved metric court frame.
    rectangle = np.array(
        [
            [-5.485, -11.885, 0],
            [5.485, -11.885, 0],
            [5.485, 11.885, 0],
            [-5.485, 11.885, 0],
            [-5.485, -11.885, 0],
        ]
    )
    for court in alignment["alignment"]["layout"]["courts"]:
        matrix = np.array(court["court_from_scene"]).reshape(4, 4)
        local = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        inside |= (np.abs(local[:, 0]) < 5.485) & (np.abs(local[:, 1]) < 11.885)
        matrix = np.array(court["scene_from_court"]).reshape(4, 4)
        corners = rectangle @ matrix[:3, :3].T + matrix[:3, 3]
        outlines.append((corners - frame["origin_metric_scene"]) @ basis)
    return {
        "uv": centered @ basis,
        "height": centered @ frame["normal_metric_scene"],
        "inside": inside,
        "outlines": outlines,
    }


def measure(
    manifest: dict, tracks: dict, points: np.ndarray, method: dict, alignment: dict
) -> dict:
    geom = geometry(points, method, alignment)
    cfg = manifest["selection"]
    eligible = (
        geom["inside"]
        & (np.abs(geom["height"]) < cfg["maximum_absolute_height_m"])
        & (tracks["reprojection_error_px"] <= cfg["maximum_reprojection_error_px"])
    )
    results = [
        paired_heights(
            geom["uv"],
            geom["height"],
            tracks["first_frame"],
            tracks["last_frame"],
            eligible,
            np.array(manifest["period_edges"]),
            size,
            cfg["minimum_points_per_cell"],
        )
        for size in (cfg["cell_size_m"], cfg["sensitivity_cell_size_m"])
    ]
    return {"primary": results[0], "sensitivity": results[1]}


def load_bundle() -> tuple[dict, dict, np.ndarray, dict, dict]:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(BUNDLE / name) != digest:
            raise ValueError(f"Changed SfM drift evidence: {name}")
    for name, digest in manifest["dependencies"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Changed SfM drift dependency: {name}")
    method = json.loads((METHOD / "manifest.json").read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    with np.load(BUNDLE / "tracks.npz") as archive:
        tracks = {k: archive[k] for k in archive.files}
    with np.load(METHOD / "arrays.npz") as archive:
        points = archive["points_xyzrgb"]
    return manifest, tracks, points, method, alignment


def collect() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    export = SCENE / "reconstruction/export"
    scene = json.loads((export / "scene.json").read_text())
    with np.load(METHOD / "arrays.npz") as archive:
        points = archive["points_xyzrgb"]
    if not np.array_equal(points, np.load(export / "points_scene.npy")):
        raise ValueError("Method point cloud differs from current B00 export")
    xyz, rgb, raw, frames = read_tracks(SCENE / "reconstruction/sfm/model")
    if frames != list(range(491)):
        raise ValueError("Expected the full saved B00 sequence, frames 0--490")
    order, error = match_export(xyz, rgb, points, np.array(scene["scene_from_sfm"]))
    np.savez_compressed(BUNDLE / "tracks.npz", **{k: v[order] for k, v in raw.items()})
    edges = np.linspace(0, len(frames), 5, dtype=int).tolist()
    photos = [f"frame_{i:06d}.png" for i in edges[:-1]]
    for name in photos:
        shutil.copyfile(export / "images" / name, BUNDLE / name)
    sources = [
        SCENE / "reconstruction/sfm/model/images.bin",
        SCENE / "reconstruction/sfm/model/points3D.bin",
        export / "scene.json",
        export / "points_scene.npy",
        *(export / "images" / name for name in photos),
    ]
    manifest = {
        "schema": "court_sfm_temporal_ground_audit_v1",
        "scene_id": "B00",
        "frame_count": len(frames),
        "point_count": len(points),
        "period_edges": edges,
        "photos": photos,
        "photo_selection": "First captured/exported RGB of each equal chronological quarter, full field of view.",
        "selection": {
            "region": "Union of interiors of the two saved doubles courts",
            "track_rule": "All observations in one quarter; boundary-crossing tracks excluded",
            "maximum_absolute_height_m": 0.3,
            "maximum_reprojection_error_px": 1.0,
            "cell_size_m": 0.5,
            "sensitivity_cell_size_m": 1.0,
            "minimum_points_per_cell": 5,
        },
        "comparison": "Median height per cell and quarter; intersect cells across all four quarters; subtract first-quarter median in each cell, then summarize across cells. Distinct points, not repeat estimates of the same point.",
        "limitations": "No survey truth or causal isolation. Surface relief within cells, feature/triangulation errors and reconstruction distortion can all contribute. Metric scale comes from the saved court alignment. B00 only, not a four-scene drift estimate.",
        "source_export_max_distance_native": error,
        "source_export_rgb_exact": True,
        "source_format": "https://colmap.github.io/format.html",
        "dependencies": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (METHOD / "manifest.json", METHOD / "arrays.npz", ALIGNMENT)
        },
        "files": {name: sha256(BUNDLE / name) for name in ["tracks.npz", *photos]},
        "source_files": {str(p.relative_to(REPO)): sha256(p) for p in sources},
    }
    write_json(BUNDLE / "manifest.json", manifest)
    write_json(BUNDLE / "measurements.json", measure(*load_bundle()))


def validate(*, check_local_sources: bool = False) -> dict:
    bundle = load_bundle()
    manifest, tracks, points, _, _ = bundle
    actual = measure(*bundle)
    if actual != json.loads((BUNDLE / "measurements.json").read_text()):
        raise ValueError("Temporal ground-height measurements differ")
    if check_local_sources:
        for name, digest in manifest["source_files"].items():
            if sha256(REPO / name) != digest:
                raise ValueError(f"Changed original SfM source: {name}")
        xyz, rgb, raw, frames = read_tracks(SCENE / "reconstruction/sfm/model")
        transform = np.array(
            json.loads((SCENE / "reconstruction/export/scene.json").read_text())[
                "scene_from_sfm"
            ]
        )
        order, error = match_export(xyz, rgb, points, transform)
        if (
            frames != list(range(manifest["frame_count"]))
            or error != manifest["source_export_max_distance_native"]
        ):
            raise ValueError("SfM source frame or point mapping differs")
        for key, value in raw.items():
            if not np.array_equal(value[order], tracks[key]):
                raise ValueError(f"Bundled track metadata differs: {key}")
    return {
        "scene_id": "B00",
        "frame_count": manifest["frame_count"],
        "point_count": len(points),
        **{
            key: {
                k: v
                for k, v in value.items()
                if k
                in ("cell_size_m", "shared_cell_count", "median_delta_m", "iqr_delta_m")
            }
            for key, value in actual.items()
        },
    }
