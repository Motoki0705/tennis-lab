"""CPU RGB/trajectory review of published SLCS pseudo teachers (not 3D GT).

Used by ``scripts.analysis.render_reconstruction_review``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tasks.slcs.data.quality import QualityConfig, build_label_masks
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.dataset_pipeline.quality import project
from src.tennis_scene.dataset_pipeline.quality_report import validate_raw_identity
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.schema import SceneResult
from src.utils.io import save_json_atomic


def validate_scene(scene: SceneResult, clip: ClipManifest) -> None:
    """Reject mismatched camera axes, calibration, dimensions and nonfinite data."""
    reference = scene.metadata["reference"]
    if list(reference["camera_ids"]) != list(clip.camera_ids):
        raise ValueError("Scene camera order differs from manifest")
    if len(clip.camera_ids) != 3 or len(reference["camera_fits"]) != 3:
        raise ValueError("Review requires exactly three reference cameras")
    if (
        (scene.num_frames, scene.width, scene.height, scene.fps)
        != (clip.num_frames, clip.width, clip.height, clip.fps)
        or not np.isfinite(scene.fps)
        or scene.fps <= 0
    ):
        raise ValueError("Scene dimensions/timeline differ from manifest")
    p, t = scene.player_position.shape[:2]
    shapes = {
        "player_position": (p, t, 3),
        "player_yaw": (p, t),
        "ball_3d": (t, 3),
        "ball_uv": (3, t, 2),
        "ball_vis": (3, t),
        "human_kp_2d": (p, 3, t, 17, 2),
        "human_kp_vis": (p, 3, t, 17),
    }
    if t != clip.num_frames or p < 1 or t < 1:
        raise ValueError("Invalid player/frame axes")
    for name, shape in shapes.items():
        array = getattr(scene, name)
        if array is None or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(
                f"Missing, nonfinite or wrong shape: {name}; expected {shape}"
            )
        if name.endswith("vis") and ((array < 0) | (array > 1)).any():
            raise ValueError(f"Invalid visibility: {name}")
    for camera in reference["camera_fits"]:
        for name, shape in (("R", (3, 3)), ("t", (3,)), ("K", (3, 3))):
            arr = np.asarray(camera[name])
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"Invalid reference camera {name}")


def label_masks(scene: SceneResult, quality: QualityConfig) -> dict[str, np.ndarray]:
    assert scene.human_kp_vis is not None and scene.ball_vis is not None
    assert scene.ball_3d is not None
    masks: dict[str, np.ndarray] = build_label_masks(
        human_kp_vis=scene.human_kp_vis,
        ball_vis=scene.ball_vis,
        player_position=scene.player_position,
        player_yaw=scene.player_yaw,
        ball_3d=scene.ball_3d,
        config=quality,
        teacher_quality=scene.metadata["label_quality"],
    )
    return masks


def ball_residual(scene: SceneResult) -> np.ndarray:
    assert (
        scene.ball_3d is not None
        and scene.ball_uv is not None
        and scene.ball_vis is not None
    )
    errors: np.ndarray = np.full(scene.ball_vis.shape, np.nan)
    for view, camera in enumerate(scene.metadata["reference"]["camera_fits"]):
        pixels, valid = project(scene.ball_3d, camera)
        valid &= scene.ball_vis[view].astype(bool)
        errors[view, valid] = np.linalg.norm(
            pixels[valid] - scene.ball_uv[view, valid] * [scene.width, scene.height],
            axis=-1,
        )
    return errors


def select_frames(
    raw: SceneResult, refined: SceneResult, masks: dict[str, np.ndarray]
) -> dict[int, list[str]]:
    """Stable first-index tie breaks; both residual extremes use one teacher mask."""
    selected: dict[int, list[str]] = {}

    def add(frame: int, reason: str) -> None:
        selected.setdefault(frame, []).append(reason)

    for frame in np.unique(np.linspace(0, refined.num_frames - 1, 5, dtype=int)):
        add(int(frame), "evenly_spaced")
    supported = masks["ball_label_weight"] > 0
    for name, scene in (("raw", raw), ("refined", refined)):
        residual = ball_residual(scene)
        valid = np.isfinite(residual) & supported[None]
        if valid.any():
            _, frame = np.unravel_index(
                np.argmax(np.where(valid, residual, -np.inf)), residual.shape
            )
            add(int(frame), f"max_{name}_ball_residual_on_positive_teacher_weight")
    support = masks["ball_label_weight"] + masks["player_label_weight"].sum(0)
    add(int(np.argmin(support)), "minimum_total_teacher_weight")
    displacement = np.linalg.norm(
        refined.player_position - raw.player_position, axis=-1
    )
    _, frame = np.unravel_index(np.argmax(displacement), displacement.shape)
    add(int(frame), "max_raw_to_refined_root_displacement")
    if refined.num_frames > 1:
        motion = np.linalg.norm(np.diff(refined.player_position, axis=1), axis=-1)
        _, frame = np.unravel_index(np.argmax(motion), motion.shape)
        add(int(frame) + 1, "max_refined_root_frame_step")
    return dict(sorted(selected.items()))


def contact_sheet(
    clip: ClipManifest,
    raw: SceneResult,
    refined: SceneResult,
    masks: dict[str, np.ndarray],
    selected: dict[int, list[str]],
    output: Path,
) -> None:
    frames = list(selected)
    fig, axes = plt.subplots(
        len(frames), 3, figsize=(18, 3.5 * len(frames)), squeeze=False
    )
    scale = np.array([clip.width, clip.height])
    assert refined.ball_uv is not None and refined.ball_vis is not None
    assert refined.human_kp_2d is not None and refined.human_kp_vis is not None
    try:
        for view, camera in enumerate(clip.camera_ids):
            capture = cv2.VideoCapture(str(clip.media_path(camera)))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open {clip.media_path(camera)}")
                for frame in range(max(frames) + 1):
                    ok, rgb = capture.read()
                    if not ok:
                        raise ValueError(f"Decode failed: {camera} frame {frame}")
                    if rgb.shape[:2] != (clip.height, clip.width):
                        raise ValueError("Source video dimensions differ from manifest")
                    if frame not in selected:
                        continue
                    ax = axes[frames.index(frame), view]
                    ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                    if refined.ball_vis[view, frame]:
                        uv = refined.ball_uv[view, frame] * scale
                        ax.scatter(*uv, c="lime", marker="o", s=90, facecolors="none")
                    for p in range(refined.player_position.shape[0]):
                        if (
                            refined.human_kp_vis[p, view, frame, [11, 12]] >= 0.3
                        ).all():
                            hip = (
                                refined.human_kp_2d[p, view, frame, [11, 12]].mean(0)
                                * scale
                            )
                            ax.scatter(*hip, c="lime", marker="+", s=100)
                            ax.annotate(f"P{p}", hip, color="lime")
                    for name, scene, color, marker in (
                        ("raw", raw, "cyan", "x"),
                        ("refined", refined, "magenta", "+"),
                    ):
                        assert scene.ball_3d is not None
                        fit = scene.metadata["reference"]["camera_fits"][view]
                        points = np.concatenate(
                            (
                                scene.ball_3d[frame : frame + 1],
                                scene.player_position[:, frame],
                            )
                        )
                        uv, front = project(points, fit)
                        weights = np.r_[
                            masks["ball_label_weight"][frame],
                            masks["player_label_weight"][:, frame],
                        ]
                        for i in np.flatnonzero(front):
                            ax.scatter(
                                *uv[i],
                                color=color if weights[i] > 0 else "orange",
                                marker=marker,
                                s=110,
                            )
                            ax.annotate(
                                f"{name} {'B' if i == 0 else 'P' + str(i - 1)}",
                                uv[i],
                                color=color if weights[i] > 0 else "orange",
                                fontsize=7,
                            )
                    ax.set_xlim(0, clip.width)
                    ax.set_ylim(clip.height, 0)
                    ax.set_title(
                        f"{camera} frame={frame} t={frame / clip.fps:.3f}s\n"
                        f"w(B,P)={np.round(np.r_[masks['ball_label_weight'][frame], masks['player_label_weight'][:, frame]], 2)}",
                        fontsize=9,
                    )
                    ax.axis("off")
            finally:
                capture.release()
        fig.suptitle(
            "PSEUDO TEACHERS, NOT measured 3D GT | observed green: ball circle / hip +\n"
            "raw cyan x; refined magenta +; zero-weight orange (shown); behind camera/offscreen excluded from image only"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output, dpi=120)
    finally:
        plt.close(fig)


def trajectory_plot(
    raw: SceneResult, refined: SceneResult, masks: dict[str, np.ndarray], output: Path
) -> None:
    assert raw.ball_3d is not None and refined.ball_3d is not None
    tracks = [("ball", raw.ball_3d, refined.ball_3d, masks["ball_label_weight"])]
    tracks += [
        (
            f"root P{p}",
            raw.player_position[p],
            refined.player_position[p],
            masks["player_label_weight"][p],
        )
        for p in range(refined.player_position.shape[0])
    ]
    fig, axes = plt.subplots(
        len(tracks), 3, figsize=(17, 3 * len(tracks)), squeeze=False
    )
    time = np.arange(refined.num_frames) / refined.fps
    try:
        for row, (name, before, after, weights) in enumerate(tracks):
            for xyz, color in enumerate(("tab:red", "tab:green", "tab:blue")):
                axes[row, 0].plot(time, before[:, xyz], "--", color=color, alpha=0.6)
                axes[row, 0].plot(time, after[:, xyz], color=color, label="XYZ"[xyz])
                axes[row, 0].scatter(
                    time[weights <= 0], after[weights <= 0, xyz], color="orange", s=8
                )
            for points, style, stage in (
                (before, "--", "raw"),
                (after, "-", "refined"),
            ):
                axes[row, 1].plot(
                    time[1:],
                    np.linalg.norm(np.diff(points, axis=0), axis=-1) * refined.fps,
                    style,
                    label=stage,
                )
            axes[row, 2].step(time, weights, where="mid", label="SLCS teacher weight")
            axes[row, 2].set_ylim(-0.05, 1.05)
            for col, unit in enumerate(("XYZ [m]", "speed [m/s]", "support [0..1]")):
                axes[row, col].set(title=f"{name} {unit}", xlabel="clip time [s]")
                axes[row, col].legend()
        fig.suptitle(
            "Pseudo labels, NOT measured 3D GT | XYZ: raw dashed / refined solid / zero weight orange\nSpeed shows every step, including unsupported endpoints"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output, dpi=130)
    finally:
        plt.close(fig)


def review(
    dataset: Path, run: Path, output: Path, clips: list[str], quality: QualityConfig
) -> None:
    output = output.resolve()
    if output.parts[-5:-2] != ("outputs", "tennis_scene", "visualize"):
        raise ValueError(
            "output-dir must be outputs/tennis_scene/visualize/<experiment>/<runid>"
        )
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not clips or len(set(clips)) != len(clips):
        raise ValueError("Choose unique clips explicitly")
    manifest = load_dataset_manifest(dataset)
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "is_ground_truth": False,
        "dataset_root": str(dataset.resolve()),
        "run_root": str(run.resolve()),
        "quality": asdict(quality),
        "clips": {},
    }
    for key in clips:
        clip = ClipManifest.load(dataset / manifest.clips[key].path)
        raw_path = run / key / "scene.npz"
        raw, refined = load_scene_result(raw_path), load_slcs_annotation(clip)
        for scene in (raw, refined):
            validate_scene(scene, clip)
        if raw.player_position.shape != refined.player_position.shape:
            raise ValueError("Raw/refined player axes differ")
        validate_raw_identity(raw, refined, refined.metadata["dataset_producer_identity"])
        masks = label_masks(refined, quality)
        selected = select_frames(raw, refined, masks)
        destination = output / key
        destination.mkdir(parents=True)
        contact_sheet(
            clip, raw, refined, masks, selected, destination / "contact_sheet.png"
        )
        trajectory_plot(raw, refined, masks, destination / "trajectory.png")
        receipt["clips"][key] = {
            "frames": [
                {"frame": f, "time_sec": f / clip.fps, "reasons": reasons}
                for f, reasons in selected.items()
            ],
            "camera_ids": list(clip.camera_ids),
            "videos": [str(clip.media_path(c).resolve()) for c in clip.camera_ids],
            "raw_scene": str(raw_path.resolve()),
            "refined_scene": str(
                (clip.clip_dir / "annotations/tennis_scene/scene.npz").resolve()
            ),
            "positive_ball_frames": int((masks["ball_label_weight"] > 0).sum()),
            "positive_player_frames": (masks["player_label_weight"] > 0)
            .sum(1)
            .tolist(),
        }
        save_json_atomic(receipt, output / "receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "run-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--min-player-confidence", type=float, default=0.3)
    parser.add_argument("--min-ball-cameras", type=int, default=1)
    parser.add_argument("--label-weight-power", type=float, default=1.0)
    args = parser.parse_args()
    review(
        args.dataset_root,
        args.run_root,
        args.output_dir,
        args.clip,
        QualityConfig(
            args.min_player_confidence,
            args.min_ball_cameras,
            args.label_weight_power,
            0.5,
        ),
    )


if __name__ == "__main__":
    main()
