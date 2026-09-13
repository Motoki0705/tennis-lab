"""Overlay both PLCS placements of the same GVHMR bodies on source videos (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.rendering.smpl_placement import place_smpl_vertices
from src.utils.rendering.mesh_renderer import MeshRenderer
from src.utils.video.writer import VideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--camera", choices=["cam0", "cam1", "cam2"], required=True)
    parser.add_argument("--preview-frame", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    scene = load_scene_result(args.comparison / "residual_scene.npz")
    if scene.smpl_vertices_local is None or scene.smpl_global_orient is None:
        raise ValueError("Cached GVHMR vertices and global orientation are required")
    with np.load(args.comparison / "baseline_best_clip.npz") as old:
        positions = np.stack([old["position"], scene.player_position])
        yaws = np.stack([old["yaw"], scene.player_yaw])
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        raise ValueError("PLCS placements must be finite")
    regressor = torch.load(args.regressor, map_location="cpu", weights_only=False)
    regressor = np.asarray(regressor, dtype=np.float32)
    mesh = MeshRenderer(load_smpl_faces(args.faces.resolve()))
    context = scene.metadata
    index = context["camera_ids"].index(args.camera)
    camera = context["camera_fits"][index]
    K, R, t = [np.array(camera[key], dtype=np.float32) for key in ["K", "R", "t"]]
    # RGB palette, shared by both players. Alpha keeps overlapping alternatives visible.
    colors = [(1.0, 0.25, 0.27), (0.05, 0.88, 1.0)]
    labels = ["Baseline epoch29", "Residual epoch4"]
    source = args.clip / "media" / f"{args.camera}.mp4"
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != scene.num_frames:
        raise ValueError("Source video and cached SMPL frame counts differ")
    if abs(cap.get(cv2.CAP_PROP_FPS) - scene.fps) > 0.01:
        raise ValueError("Source video and cached SMPL frame rates differ")
    output = args.comparison / "smpl_overlay"
    output.mkdir(exist_ok=True)
    video = output / f"{args.camera}_comparison.mp4"
    indices = range(scene.num_frames)
    if args.preview_frame is not None:
        if not 0 <= args.preview_frame < scene.num_frames:
            raise ValueError("Preview frame outside clip")
        indices = range(args.preview_frame, args.preview_frame + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
    writer = VideoWriter(video, fps=scene.fps) if args.preview_frame is None else None
    try:
        for frame_index in indices:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_index}")
            if frame.shape[:2] != (scene.height, scene.width):
                raise ValueError("Source dimensions differ from calibration")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bodies = []
            for model in range(2):
                sl = slice(frame_index, frame_index + 1)
                world = place_smpl_vertices(
                    scene.smpl_vertices_local[:, sl],
                    scene.smpl_global_orient[:, sl],
                    positions[model, :, sl],
                    yaws[model, :, sl],
                    regressor,
                )[:, 0]
                camera_vertices = world @ R.T + t
                for player, vertices in enumerate(camera_vertices):
                    bodies.append(
                        (float(vertices[:, 2].mean()), model, player, vertices)
                    )
            # Far bodies first; the established renderer depth-sorts faces within each mesh.
            for _, model, _, vertices in sorted(bodies, key=lambda body: -body[0]):
                image = mesh.render_overlay(
                    image, vertices, K, color=colors[model], alpha=0.62
                )
            cv2.rectangle(image, (0, 0), (scene.width, 90), (18, 22, 28), -1)
            for model, label in enumerate(labels):
                color = tuple(int(v * 255) for v in colors[model])
                cv2.putText(
                    image,
                    label,
                    (24 + model * 460, 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                f"{args.camera} | {frame_index / scene.fps:.2f}s | Same GVHMR pose, two PLCS placements per player",
                (24, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            if writer is not None:
                writer.write_frame(image)
            if args.preview_frame is not None or frame_index in [
                0,
                250,
                500,
                750,
                1009,
            ]:
                cv2.imwrite(
                    str(output / f"{args.camera}_{frame_index:04d}.jpg"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
            if frame_index % 100 == 0:
                print(
                    f"{args.camera}: {frame_index + 1}/{scene.num_frames}", flush=True
                )
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if writer is not None:
        receipt = {
            "source": str(source.resolve()),
            "output": str(video.resolve()),
            "frames": scene.num_frames,
            "fps": scene.fps,
            "width": scene.width,
            "height": scene.height,
            "meshes_per_frame": int(positions.shape[0] * positions.shape[1]),
            "placements": labels,
            "alpha": 0.62,
            "smpl_source": str((args.comparison / "residual_scene.npz").resolve()),
            "faces": str(args.faces.resolve()),
            "regressor": str(args.regressor.resolve()),
            "camera": camera,
            "transform": "pelvis center -> undo GVHMR global orient -> Y-up to Z-up -> PLCS yaw -> PLCS position",
        }
        (output / f"{args.camera}_comparison.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
