"""Audit temporal ground-height disagreement in all four saved sparse SfM models.

COLMAP's documented binary tracks identify observation frames, not acquisition
times of the reconstructed coordinates. Distinct short tracks are compared in
shared ground cells; these measurements are not absolute drift ground truth.
Binary format: https://colmap.github.io/format.html
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from common import REPO, ROOT, sha256, write_json
from scipy.spatial import cKDTree

BUNDLE = ROOT / "evidence/sfm_drift"
METHOD = ROOT / "evidence/alignment_method"
SCENE_IDS = ("B00", "B01", "B02", "B03")
SCENES = REPO / "data/synthetic_data_generation/scenes"
SELECTION = {
    "region": "Union of interiors of all saved doubles courts in each scene",
    "track_rule": "All observations in one period; boundary-crossing tracks excluded",
    "maximum_absolute_height_m": 0.3,
    "maximum_reprojection_error_px": 1.0,
    "cell_size_m": 0.5,
    "sensitivity_cell_size_m": 1.0,
    "minimum_points_per_cell": 5,
}


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
                raise ValueError(f"Unexpected frame name: {stem}")
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
    result = {
        "status": "measured" if shared else "no_shared_cells",
        "cell_size_m": cell_size,
        "qualified_point_counts": point_counts,
        "qualified_cell_counts": cell_counts,
        "shared_cell_count": len(shared),
        "shared_cell_indices": np.array(shared).tolist(),
    }
    if not shared:
        # An unsupported comparison is not a zero-error measurement.
        return {
            **result,
            "cell_median_height_m": [],
            "cell_delta_from_first_m": [],
            "median_delta_m": None,
            "iqr_delta_m": None,
        }
    medians = np.array([[p[cell] for cell in shared] for p in per_period])
    delta = medians - medians[0]
    return {
        **result,
        "cell_median_height_m": medians.tolist(),
        "cell_delta_from_first_m": delta.tolist(),
        "median_delta_m": np.median(delta, axis=1).tolist(),
        "iqr_delta_m": np.quantile(delta, [0.25, 0.75], axis=1).T.tolist(),
    }


def period_edges(frames: list[int], periods: int) -> list[int]:
    """Split registered images by chronological rank, retaining gaps and the tail."""
    if frames != sorted(set(frames)) or len(frames) < periods:
        raise ValueError("Expected unique chronological registered frames")
    indices = np.linspace(0, len(frames), periods + 1, dtype=int)
    return [frames[i] for i in indices[:-1]] + [frames[-1] + 1]


def ground_reference(alignment: dict) -> dict:
    """Use the same saved-court-plane construction, including manual scenes."""
    courts = alignment["alignment"]["layout"]["courts"]
    matrix = np.array(courts[0]["scene_from_court"]).reshape(4, 4)
    normal = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    height = float(normal @ matrix[:3, 3])
    for court in courts:
        matrix = np.array(court["scene_from_court"]).reshape(4, 4)
        if (
            not np.allclose(matrix[:3, 2], normal, atol=1e-8, rtol=0)
            or abs(float(normal @ matrix[:3, 3]) - height) > 1e-8
        ):
            raise ValueError("Saved courts do not share a common ground plane")
    u = np.array([1.0, 0, 0]) - normal * normal[0]
    if np.linalg.norm(u) < 0.1:
        raise ValueError("Ground U basis is degenerate")
    u /= np.linalg.norm(u)
    return {
        "authority": "Common plane of saved aligned courts; closest origin to metric scene origin; U is projected scene X, V = normal cross U",
        "nht_scene_units_per_metre": alignment["alignment"]["metric_scene_adapter"][
            "nht_scene_units_per_metre"
        ],
        "ground_plane_frame": {
            "origin_metric_scene": (normal * height).tolist(),
            "normal_metric_scene": normal.tolist(),
            "basis_u_metric_scene": u.tolist(),
            "basis_v_metric_scene": np.cross(normal, u).tolist(),
        },
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
    results = {}
    for key, edges, size in (
        ("primary", manifest["period_edges"], cfg["cell_size_m"]),
        ("sensitivity", manifest["period_edges"], cfg["sensitivity_cell_size_m"]),
        ("four_period_audit", manifest["four_period_edges"], cfg["cell_size_m"]),
        (
            "four_period_sensitivity",
            manifest["four_period_edges"],
            cfg["sensitivity_cell_size_m"],
        ),
    ):
        results[key] = paired_heights(
            geom["uv"],
            geom["height"],
            tracks["first_frame"],
            tracks["last_frame"],
            eligible,
            np.array(edges),
            size,
            cfg["minimum_points_per_cell"],
        )
    if (
        results["primary"]["status"] != "measured"
        or results["sensitivity"]["status"] != "measured"
    ):
        raise ValueError("Primary half-sequence comparison has no shared support")
    return results


def load_bundle(scene_id: str) -> tuple[dict, dict, np.ndarray, dict, dict]:
    index = json.loads((BUNDLE / "manifest.json").read_text())
    if (
        index["schema"] != "court_sfm_temporal_ground_audit_v2"
        or tuple(index["scenes"]) != SCENE_IDS
    ):
        raise ValueError("Expected SfM evidence for all four scenes")
    manifest = {"selection": index["selection"], **index["scenes"][scene_id]}
    for name, digest in manifest["files"].items():
        if sha256(BUNDLE / name) != digest:
            raise ValueError(f"Changed SfM drift evidence: {name}")
    for name, digest in manifest["dependencies"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Changed SfM drift dependency: {name}")
    alignment = json.loads(
        (ROOT / f"evidence/scene_sources/{scene_id}.json").read_text()
    )
    reference = ground_reference(alignment)
    if reference != manifest["ground_reference"]:
        raise ValueError("Saved-court ground reference differs")
    frames = manifest["registered_frame_ids"]
    if manifest["period_edges"] != period_edges(frames, 2) or manifest[
        "four_period_edges"
    ] != period_edges(frames, 4):
        raise ValueError("Chronological period boundaries differ")
    with np.load(BUNDLE / scene_id / "tracks.npz") as archive:
        tracks = {k: archive[k] for k in archive.files}
    with np.load(ROOT / manifest["points_archive"]) as archive:
        points = archive["points_xyzrgb"]
    return manifest, tracks, points, reference, alignment


def collect() -> None:
    index = {
        "schema": "court_sfm_temporal_ground_audit_v2",
        "selection": SELECTION,
        "protocol": "All scenes: first/second chronological halves of registered frames. Four-period comparisons retained as an audit, explicitly unsupported when no cell is shared. The original four-period 0.5 m rule has no support in B01--B03; neither cell width nor minimum support is tuned per scene.",
        "comparison": "Median height per cell and period; intersect cells across periods; subtract the first-period median in each cell, then summarize equally weighted cells. Distinct points, not repeat estimates of the same point.",
        "limitations": "No survey truth or causal isolation. Surface relief within cells, feature/triangulation errors and reconstruction distortion can all contribute. Metric scale and plane come from saved court alignment, including manual B01/B02 adjustments. Missing registered frames are not interpolated.",
        "scenes": {},
    }
    for sid in SCENE_IDS:
        bundle = BUNDLE / sid
        bundle.mkdir(parents=True, exist_ok=True)
        source = SCENES / sid
        export = source / "reconstruction/export"
        scene = json.loads((export / "scene.json").read_text())
        cameras = json.loads((export / "cameras.json").read_text())["cameras"]
        alignment_path = ROOT / f"evidence/scene_sources/{sid}.json"
        alignment = json.loads(alignment_path.read_text())
        if sha256(source / "alignment/alignment.json") != alignment["alignment_sha256"]:
            raise ValueError(f"Saved alignment changed: {sid}")
        points = np.load(export / "points_scene.npy")
        points_path = METHOD / "arrays.npz" if sid == "B00" else bundle / "points.npz"
        if sid == "B00":
            with np.load(points_path) as archive:
                if not np.array_equal(points, archive["points_xyzrgb"]):
                    raise ValueError(
                        "Method point cloud differs from current B00 export"
                    )
        else:
            np.savez_compressed(points_path, points_xyzrgb=points)
        xyz, rgb, raw, frames = read_tracks(source / "reconstruction/sfm/model")
        exported_frames = sorted(int(c["camera_id"][6:]) for c in cameras)
        if frames != exported_frames or len(frames) != scene["camera_count"]:
            raise ValueError(f"SfM and exported registered cameras differ: {sid}")
        order, error = match_export(xyz, rgb, points, np.array(scene["scene_from_sfm"]))
        np.savez_compressed(
            bundle / "tracks.npz", **{k: v[order] for k, v in raw.items()}
        )
        sources = [
            source / "reconstruction/sfm/model/images.bin",
            source / "reconstruction/sfm/model/points3D.bin",
            export / "scene.json",
            export / "cameras.json",
            export / "points_scene.npy",
            source / "alignment/alignment.json",
        ]
        files = [bundle / "tracks.npz"] + ([] if sid == "B00" else [points_path])
        dependencies = [alignment_path] + ([points_path] if sid == "B00" else [])
        index["scenes"][sid] = {
            "scene_id": sid,
            "frame_count": len(frames),
            "point_count": len(points),
            "registered_frame_ids": frames,
            "unregistered_frame_ids_within_span": sorted(
                set(range(frames[0], frames[-1] + 1)) - set(frames)
            ),
            "period_edges": period_edges(frames, 2),
            "four_period_edges": period_edges(frames, 4),
            "ground_reference": ground_reference(alignment),
            "points_archive": str(points_path.relative_to(ROOT)),
            "source_export_max_distance_native": error,
            "source_export_rgb_exact": True,
            "source_format": "https://colmap.github.io/format.html",
            "dependencies": {str(p.relative_to(ROOT)): sha256(p) for p in dependencies},
            "files": {str(p.relative_to(BUNDLE)): sha256(p) for p in files},
            "source_files": {str(p.relative_to(REPO)): sha256(p) for p in sources},
        }
    write_json(BUNDLE / "manifest.json", index)
    write_json(
        BUNDLE / "measurements.json",
        {sid: measure(*load_bundle(sid)) for sid in SCENE_IDS},
    )


def validate(*, check_local_sources: bool = False) -> dict:
    saved = json.loads((BUNDLE / "measurements.json").read_text())
    if tuple(saved) != SCENE_IDS:
        raise ValueError("Missing scene in SfM measurements")
    return {
        sid: validate_scene(sid, saved[sid], check_local_sources=check_local_sources)
        for sid in SCENE_IDS
    }


def validate_scene(scene_id: str, saved: dict, *, check_local_sources: bool) -> dict:
    bundle = load_bundle(scene_id)
    manifest, tracks, points, _, _ = bundle
    actual = measure(*bundle)
    if actual != saved:
        raise ValueError("Temporal ground-height measurements differ")
    if check_local_sources:
        for name, digest in manifest["source_files"].items():
            if sha256(REPO / name) != digest:
                raise ValueError(f"Changed original SfM source: {name}")
        source = SCENES / scene_id
        xyz, rgb, raw, frames = read_tracks(source / "reconstruction/sfm/model")
        transform = np.array(
            json.loads((source / "reconstruction/export/scene.json").read_text())[
                "scene_from_sfm"
            ]
        )
        order, error = match_export(xyz, rgb, points, transform)
        if (
            frames != manifest["registered_frame_ids"]
            or error != manifest["source_export_max_distance_native"]
        ):
            raise ValueError("SfM source frame or point mapping differs")
        for key, value in raw.items():
            if not np.array_equal(value[order], tracks[key]):
                raise ValueError(f"Bundled track metadata differs: {key}")
    return {
        "scene_id": scene_id,
        "frame_count": manifest["frame_count"],
        "point_count": len(points),
        **{
            key: {
                k: v
                for k, v in value.items()
                if k
                in (
                    "status",
                    "cell_size_m",
                    "shared_cell_count",
                    "median_delta_m",
                    "iqr_delta_m",
                )
            }
            for key, value in actual.items()
        },
    }
