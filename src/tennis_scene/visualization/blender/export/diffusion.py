"""Numerical assets for a schematic trajectory-completion concept, not inference."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator, make_smoothing_spline

from src.tennis_scene.visualization.blender.export.pointcache import write_pc2

SCHEMA = "tennis_scene_ball_diffusion_concept_v1"


@dataclass(frozen=True)
class DiffusionConfig:
    seed: int
    smoothing_lambda: float
    point_tolerance_m: float
    noise_knots: int
    noise_scale_m: tuple[float, ...]
    noise_center_m: tuple[float, ...]
    fps: int
    conditions_seconds: float
    denoise_seconds: float
    playback_seconds: float
    curve_radius_m: float
    tube_sides: int
    condition_radius_m: float
    gap_radius_m: float
    ball_radius_m: float
    width: int
    height: int
    samples: int
    gif_width: int
    gif_fps: int

    def __post_init__(self) -> None:
        integers = (
            self.seed,
            self.noise_knots,
            self.fps,
            self.tube_sides,
            self.width,
            self.height,
            self.samples,
            self.gif_width,
            self.gif_fps,
        )
        if any(type(x) is not int for x in integers) or self.seed < 0:
            raise ValueError(
                "Require integer seed >= 0 and integer render dimensions/counts"
            )
        if min(integers[1:]) < 1 or self.noise_knots < 4 or self.tube_sides < 4:
            raise ValueError("Require positive counts, noise_knots/tube_sides >= 4")
        scalars = (
            self.smoothing_lambda,
            self.point_tolerance_m,
            self.conditions_seconds,
            self.denoise_seconds,
            self.playback_seconds,
            self.curve_radius_m,
            self.condition_radius_m,
            self.gap_radius_m,
            self.ball_radius_m,
        )
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError(
                "Require finite positive smoothing, tolerance, durations and radii"
            )
        if any(len(x) != 3 for x in (self.noise_scale_m, self.noise_center_m)):
            raise ValueError("noise_scale_m/noise_center_m must have three components")
        if (
            not np.isfinite((*self.noise_scale_m, *self.noise_center_m)).all()
            or min(self.noise_scale_m) <= 0
        ):
            raise ValueError("Noise vectors must be finite, with positive scales")
        if any(
            round(s * self.fps) < 2
            for s in (
                self.conditions_seconds,
                self.denoise_seconds,
                self.playback_seconds,
            )
        ):
            raise ValueError("Each stage must contain at least two output frames")
        if (
            self.width % 2
            or self.height % 2
            or self.gif_width % 2
            or self.gif_fps > self.fps
        ):
            raise ValueError("Require even image dimensions and gif_fps <= fps")


def smooth_trajectory(
    positions: np.ndarray, *, fps: float, smoothing_lambda: float, tolerance_m: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit cubic smoothing splines with a bounded residual at every valid point.

    Reduce the requested lambda by bisection until ALL observed 3D residuals
    meet the bound. Interior gaps are evaluated from that same fitted spline.
    No point rejection, physical simulation, or model inference is performed.
    """
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (T,3)")
    valid = np.isfinite(points).all(axis=1)
    if np.isinf(points).any() or np.any(np.isfinite(points).any(axis=1) != valid):
        raise ValueError(
            "Invalid positions must be all NaN, not partial/non-finite coordinates"
        )
    if valid.sum() < 5 or not valid[0] or not valid[-1]:
        raise ValueError(
            "Need >= 5 valid points and observed endpoints; extrapolation is forbidden"
        )
    if (
        not np.isfinite((fps, smoothing_lambda, tolerance_m)).all()
        or min(fps, smoothing_lambda, tolerance_m) <= 0
    ):
        raise ValueError(
            "Require finite positive fps, smoothing_lambda and tolerance_m"
        )
    time = np.arange(len(points), dtype=np.float64) / fps

    def fit(lam: float) -> np.ndarray:
        return cast(
            np.ndarray,
            np.column_stack(
                [
                    make_smoothing_spline(time[valid], points[valid, axis], lam=lam)(
                        time
                    )
                    for axis in range(3)
                ]
            ),
        )

    best, chosen = fit(0.0), 0.0
    low, high = 0.0, smoothing_lambda
    candidate = fit(high)
    if np.linalg.norm(candidate[valid] - points[valid], axis=1).max() <= tolerance_m:
        best, chosen = candidate, high
    else:
        for _ in range(32):
            lam = (low + high) / 2
            candidate = fit(lam)
            if (
                np.linalg.norm(candidate[valid] - points[valid], axis=1).max()
                <= tolerance_m
            ):
                best, chosen, low = candidate, lam, lam
            else:
                high = lam
    error = np.linalg.norm(best[valid] - points[valid], axis=1)
    if not np.isfinite(best).all() or error.max() > tolerance_m + 1e-9:
        raise ValueError("Schematic spline did not satisfy the point tolerance")
    return best, {
        "method": "per-axis cubic smoothing spline; lambda reduced by 32 bisections to bound every valid 3D residual",
        "objective": "sum_i (y_i-f(t_i))^2 + lambda * integral (f''(t))^2 dt; times in seconds",
        "requested_lambda": smoothing_lambda,
        "effective_lambda": chosen,
        "point_tolerance_m": tolerance_m,
        "max_residual_m": float(error.max()),
        "rms_residual_m": float(np.sqrt(np.mean(error**2))),
        "valid_points": int(valid.sum()),
        "filled_frames": np.flatnonzero(~valid).tolist(),
        "edge_policy": "reject unobserved endpoints; no extrapolation",
        "model_output": False,
        "physics_model": False,
    }


def concept_arrays(
    positions: np.ndarray, source_fps: float, config: DiffusionConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    trajectory, fit = smooth_trajectory(
        positions,
        fps=source_fps,
        smoothing_lambda=config.smoothing_lambda,
        tolerance_m=config.point_tolerance_m,
    )
    rng = np.random.default_rng(config.seed)
    controls = np.asarray(config.noise_center_m) + np.asarray(
        config.noise_scale_m
    ) * np.tanh(rng.standard_normal((config.noise_knots, 3)))
    noise = PchipInterpolator(
        np.linspace(0, len(positions) - 1, config.noise_knots), controls
    )(np.arange(len(positions)))
    counts = [
        round(s * config.fps)
        for s in (
            config.conditions_seconds,
            config.denoise_seconds,
            config.playback_seconds,
        )
    ]
    a, b, c = counts
    progress = np.linspace(0, 1, b)
    alpha = np.concatenate((np.zeros(a), progress**2 * (3 - 2 * progress), np.ones(c)))
    process = (1 - alpha[:, None, None]) * noise + alpha[:, None, None] * trajectory
    # Keep exact endpoints rather than relying on floating point cancellation.
    process[alpha == 0], process[alpha == 1] = noise, trajectory
    playback_index = np.concatenate(
        (np.zeros(a + b), np.linspace(0, len(positions) - 1, c))
    )
    ball = np.column_stack(
        [
            np.interp(playback_index, np.arange(len(positions)), trajectory[:, axis])
            for axis in range(3)
        ]
    )
    return {
        "trajectory": trajectory,
        "noise": noise,
        "process": process,
        "alpha": alpha,
        "ball_position": ball,
        "ball_visible": np.arange(sum(counts)) >= a + b,
        "curve_visible": np.arange(sum(counts)) >= a,
    }, {
        "fit": fit,
        "stages": [
            {
                "name": name,
                "output_frames": [sum(counts[:i]), sum(counts[: i + 1])],
                "seconds": counts[i] / config.fps,
            }
            for i, name in enumerate(("conditions", "denoise", "playback"))
        ],
        "playback": {
            "source_frame_offsets": playback_index.tolist(),
            "speed": ((len(positions) - 1) / source_fps) / ((c - 1) / config.fps),
            "sampling": "linear interpolation of schematic all-frame trajectory; inclusive endpoints",
        },
        "noise": {
            "seed": config.seed,
            "generator": "numpy.default_rng PCG64",
            "method": "independent standard normal control points, tanh bounded, PCHIP in source time",
            "schedule": "(1-smoothstep(u))*noise + smoothstep(u)*trajectory; u spans denoise stage",
        },
    }


def tube_geometry(
    process: np.ndarray, *, radius: float, sides: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export a tube plus unconnected center vertices for evaluated validation."""
    tangent = np.gradient(process, axis=1)
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    # A stationary point has no tangent. Choose a documented display orientation.
    tangent = np.divide(
        tangent,
        length,
        out=np.broadcast_to([1.0, 0.0, 0.0], tangent.shape).copy(),
        where=length > 1e-10,
    )
    axis = np.eye(3)[np.argmin(np.abs(tangent), axis=-1)]
    side = np.cross(tangent, axis)
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    up = np.cross(tangent, side)
    angles = np.arange(sides) * (2 * math.pi / sides)
    rings = process[:, :, None, :] + radius * (
        np.cos(angles)[None, None, :, None] * side[:, :, None, :]
        + np.sin(angles)[None, None, :, None] * up[:, :, None, :]
    )
    vertices = np.concatenate((rings, process[:, :, None, :]), axis=2).reshape(
        len(process), -1, 3
    )
    faces = np.array(
        [
            (
                i * (sides + 1) + j,
                i * (sides + 1) + (j + 1) % sides,
                (i + 1) * (sides + 1) + (j + 1) % sides,
                (i + 1) * (sides + 1) + j,
            )
            for i in range(process.shape[1] - 1)
            for j in range(sides)
        ],
        dtype=np.int32,
    )
    centers: np.ndarray = np.arange(process.shape[1]) * (sides + 1) + sides
    return vertices.astype(np.float32), faces, centers


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _badge(path: Path, font: Path) -> None:
    image = Image.new("RGBA", (920, 138), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (4, 4, 910, 130),
        radius=22,
        fill=(19, 24, 35, 245),
        outline=(188, 143, 255, 255),
        width=3,
    )
    draw.text(
        (30, 12),
        "構想",
        font=ImageFont.truetype(str(font), 52),
        fill=(221, 198, 255, 255),
    )
    face = ImageFont.truetype(str(font), 26)
    draw.text((200, 25), "三角測量点を条件に、軌道を補完", font=face, fill="white")
    legend_font = ImageFont.truetype(str(font), 23)
    for x, color, label, hollow in (
        (30, "#4CC9F0", "三角測量", False),
        (230, "#E63946", "外れ観測を含む", False),
        (520, "#FFD166", "欠測（位置は模式）", True),
    ):
        draw.ellipse(
            (x, 96, x + 16, 112), fill=None if hollow else color, outline=color, width=3
        )
        draw.text((x + 28, 88), label, font=legend_font, fill=(180, 210, 220, 255))
    image.save(path)


def export_diffusion_bundle(
    delivery: Path,
    output: Path,
    *,
    font: Path,
    config: DiffusionConfig,
    resolved_config: dict[str, Any],
) -> Path:
    """Read a pinned rays delivery; write new numerical assets atomically."""
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}")
    source = delivery / "bundle"
    manifest = json.loads((source / "manifest.json").read_text())
    shot = json.loads((delivery / "shot/shot_manifest.json").read_text())
    if manifest["schema"] != "tennis_scene_blender_rays_v1":
        raise ValueError("Unsupported source rays schema")
    start, stop = manifest["frames"]
    positions = np.load(source / "ball_position.npy", allow_pickle=False)
    reasons = np.load(source / "ball_reason.npy", allow_pickle=False)
    states = np.load(source / "ray_state.npy", allow_pickle=False)
    valid = np.isfinite(positions).all(axis=1)
    if (
        positions.shape != (stop - start, 3)
        or reasons.shape != (stop - start,)
        or states.shape != (len(manifest["cameras"]), stop - start)
    ):
        raise ValueError("Source bundle array shapes disagree with frames/cameras")
    if (
        not np.array_equal(valid, reasons == 0)
        or not np.isin(states, np.arange(5)).all()
    ):
        raise ValueError("Source validity/reasons/ray states disagree")
    if (
        shot["ball_source"] != manifest["ball_source"]
        or shot["source_frames"][-1] != stop - 1
    ):
        raise ValueError(
            "Source shot must end at the bundle endpoint and preserve ball_source"
        )
    arrays, details = concept_arrays(positions, manifest["fps"], config)
    arrays["condition_position"], arrays["condition_reason"] = positions, reasons
    arrays["condition_valid"] = valid
    details["fit"]["filled_source_frames"] = (start + np.flatnonzero(~valid)).tolist()
    arrays["condition_state"] = np.where(
        valid, np.where((states == 2).any(axis=0), 2, 1), 0
    ).astype(np.uint8)
    vertices, faces, centers = tube_geometry(
        arrays["process"], radius=config.curve_radius_m, sides=config.tube_sides
    )
    origins = np.load(source / "ray_origin.npy", allow_pickle=False)
    if origins.shape != (len(manifest["cameras"]), 3) or not np.isfinite(origins).all():
        raise ValueError("Invalid source camera centers")
    selected = next(
        c
        for c in manifest["cameras"]
        if c["id"] == shot["camera_selection"]["camera_id"]
    )
    angle = math.atan2(selected["center"][1], selected["center"][0])
    radius = max(45.0, float(np.linalg.norm(origins[:, :2], axis=1).max()) + 30.0)
    location = np.array(
        [radius * math.cos(angle), radius * math.sin(angle), radius * 0.7]
    )
    forward = np.array([0.0, 0.0, 1.0]) - location
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    camera_matrix = np.eye(4)
    camera_matrix[:3, :3], camera_matrix[:3, 3] = rotation, location
    input_paths = [
        delivery / "manifest.json",
        source / "manifest.json",
        delivery / "shot/shot_manifest.json",
        delivery / "shot/ball_rays.blend",
        source / "ball_position.npy",
        source / "ball_reason.npy",
        source / "ray_state.npy",
        source / "ray_origin.npy",
        font,
    ]
    # Copy just the final plates. The new scene freezes the preceding shot's view.
    plates = [
        source / c["plate_dir"] / f"{stop - 1:05d}.jpg" for c in manifest["cameras"]
    ]
    input_paths.extend(plates)
    inputs = [
        {"path": str(path.resolve()), "sha256": _sha(path)} for path in input_paths
    ]
    result = {
        "schema": SCHEMA,
        "source_frames": [start, stop],
        "source_fps": manifest["fps"],
        "fps": config.fps,
        "num_frames": len(arrays["alpha"]),
        "config": asdict(config),
        "resolved_config": resolved_config,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "inputs": inputs,
        "ball_source": manifest["ball_source"],
        "source_provenance": manifest["provenance"],
        "origins": {
            "conditions": "measured",
            "process": "schematic",
            "trajectory": "schematic",
            "overall": "concept",
            "gap_markers": "schematic",
        },
        "condition_states": {"invalid": 0, "valid": 1, "valid_with_outlier_view": 2},
        "gap_marker_policy": "No measured 3D position exists. Hollow amber markers use fitted schematic positions; original NaNs/reasons are preserved.",
        "tube_orientation_policy": "minimum-component cross axis; stationary tangent uses +X for display only",
        "camera_continuity": {
            "policy": "copy evaluated final state of source blend; verify against source shot overview formula",
            "source_output_frame": len(shot["source_frames"]) - 1,
            "matrix_world": camera_matrix.tolist(),
            "lens_mm": 32.0,
            "sensor_width_mm": 36.0,
            "source_shot": shot["shot"],
            "source_blend_sha256": inputs[3]["sha256"],
        },
        "cameras": [
            {"id": c["id"], "center": c["center"], "plate": f"plate_{i}.jpg"}
            for i, c in enumerate(manifest["cameras"])
        ],
        **details,
    }
    output.mkdir(parents=True)
    try:
        for name, array in arrays.items():
            np.save(output / f"{name}.npy", array, allow_pickle=False)
        np.save(output / "tube_faces.npy", faces, allow_pickle=False)
        np.save(output / "tube_center_indices.npy", centers, allow_pickle=False)
        write_pc2(output / "process.pc2", vertices, start_frame=0.0)
        # HUD quad in camera-local coordinates, 1m ahead of the 32mm camera.
        half_width = 36.0 / (2 * 32.0)
        half_height = half_width * config.height / config.width
        left, top = -half_width * 0.94, half_height * 0.93
        right = left + 2 * half_width * 920 / 1920
        bottom = top - 2 * half_height * 138 / 1080
        np.save(
            output / "badge_vertices.npy",
            [
                [left, bottom, -1.0],
                [right, bottom, -1.0],
                [right, top, -1.0],
                [left, top, -1.0],
            ],
            allow_pickle=False,
        )
        _badge(output / "badge.png", font)
        for i, plate in enumerate(plates):
            shutil.copyfile(plate, output / f"plate_{i}.jpg")
        result["assets"] = {p.name: _sha(p) for p in sorted(output.iterdir())}
        (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
