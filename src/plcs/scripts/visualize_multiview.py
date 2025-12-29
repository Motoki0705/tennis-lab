"""Visualize PLCS multi-view scenes and run model predictions (Hydra-based).

Example commands:
    `uv run python -m src.plcs.scripts.visualize_multiview`
    `uv run python -m src.plcs.scripts.visualize_multiview \
        visualization.scene_path=data/plcs/scenes/scene_000000.npz \
        visualization.mode=predict \
        visualization.checkpoint=outputs/plcs_multiview/checkpoints/last.ckpt`

Config entry point: `src/plcs/configs/visualize_multiview.yaml`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.plcs.generate_dataset.io.dataset_io import load_scene
from src.utils.rendering import PLCSSceneRenderer as SceneRenderer

if TYPE_CHECKING:
    from src.plcs.generate_dataset.scene_generator import SceneData


@dataclass
class RuntimeConfig:
    """Resolved configuration values for visualization workflows."""

    mode: str
    scene_path: Path
    frame: int
    view: str
    cameras: list[int]
    animation_view: str
    fps: float | None
    save: Path | None
    save_input: Path | None
    info: bool
    checkpoint: str | None
    device: str


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Convert Hydra config into runtime-friendly values."""
    vis = cfg.visualization

    # Parse cameras: can be "0,1,2" or [0, 1, 2] or "all"
    cameras_raw = vis.get("cameras", "all")
    if cameras_raw == "all":
        cameras = []  # Will be filled based on scene
    elif isinstance(cameras_raw, str):
        cameras = [int(c.strip()) for c in cameras_raw.split(",")]
    else:
        cameras = list(cameras_raw)

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        frame=int(vis.frame),
        view=str(vis.view),
        cameras=cameras,
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        save_input=(
            Path(to_absolute_path(str(vis.save_input))) if vis.save_input else None
        ),
        info=bool(vis.info),
        checkpoint=(
            to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None
        ),
        device=_resolve_device(str(vis.device)),
    )


def _require_checkpoint(cfg: RuntimeConfig) -> bool:
    if cfg.checkpoint is None:
        print("Error: checkpoint must be provided for prediction modes.")
        return False
    return True


def print_scene_info(scene: SceneData) -> None:
    """Print scene metadata and statistics."""
    meta = scene.meta
    print("=" * 60)
    print("Scene Information (Multi-View)")
    print("=" * 60)
    print(f"Scene ID:        {meta['scene_id']}")
    print(f"Motion source:   {meta['motion_source']}")
    print(f"Category:        {meta['motion_category']}")
    print(f"Gender:          {meta['gender']}")
    print(f"FPS:             {meta['fps']}")
    print(f"Num frames:      {meta['num_frames']}")
    print(f"Duration:        {meta['num_frames'] / meta['fps']:.2f} seconds")
    print(
        f"Initial pos:     ({meta['initial_position'][0]:.2f}, "
        f"{meta['initial_position'][1]:.2f})"
    )
    print(f"Initial yaw:     {np.degrees(meta['initial_yaw']):.1f}°")
    print(f"Cameras sampled: {meta['num_cameras_sampled']}")
    num_cameras = len(scene.cameras)
    print(f"Cameras available: {num_cameras}")
    print()
    print("Position statistics (normalized):")
    print(
        f"  X range: [{scene.position[:, 0].min():.3f}, "
        f"{scene.position[:, 0].max():.3f}]"
    )
    print(
        f"  Y range: [{scene.position[:, 1].min():.3f}, "
        f"{scene.position[:, 1].max():.3f}]"
    )


def validate_frame_and_cameras(
    scene: SceneData, cfg: RuntimeConfig
) -> tuple[int | None, list[int]]:
    """Validate frame and camera indices. Returns error code and resolved cameras."""
    num_frames = scene.meta["num_frames"]
    if cfg.frame < 0 or cfg.frame >= num_frames:
        print(f"Error: Frame {cfg.frame} out of range (0-{num_frames - 1})")
        return 1, []

    num_cameras = len(scene.cameras)
    if not cfg.cameras:
        cameras = list(range(num_cameras))
    else:
        cameras = cfg.cameras
        for cam in cameras:
            if cam < 0 or cam >= num_cameras:
                print(f"Error: Camera {cam} out of range (0-{num_cameras - 1})")
                return 1, []

    if len(cameras) < 2:
        print("Warning: Multi-view prediction works best with >= 2 cameras")

    return None, cameras


def render_scene(scene: SceneData, cfg: RuntimeConfig) -> int:
    """Render a scene with the configured view settings."""
    renderer = SceneRenderer()

    if cfg.view == "animation":
        meta = scene.meta
        fps = cfg.fps or float(meta.get("fps", 30.0))
        print(f"Creating animation ({cfg.animation_view} view)...")
        camera_idx = cfg.cameras[0] if cfg.cameras else 0
        anim = renderer.create_animation(
            scene,
            view=cfg.animation_view,
            camera_idx=camera_idx,
            fps=fps,
        )

        if anim is None:
            return 1

        if cfg.save:
            cfg.save.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving animation to {cfg.save}...")
            anim.save(str(cfg.save), fps=fps)
            plt.close()
            print("Done!")
        else:
            plt.show()

    elif cfg.view == "3d":
        print(f"Rendering 3D view (frame {cfg.frame})...")
        render_3d = getattr(renderer, "render_3d_view", None)
        if render_3d is None:
            render_3d = (
                getattr(renderer, "render_3d", None)
                or getattr(renderer, "render_scene_3d", None)
                or getattr(renderer, "render_view_3d", None)
            )
        if render_3d is None:
            fig, _ax = plt.subplots()
            _ax.set_axis_off()
            _ax.set_title("3D view unavailable")
        else:
            try:
                fig, _ax = render_3d(scene, cfg.frame)
            except TypeError:
                fig, _ax = render_3d(scene)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "multi":
        print(f"Rendering multi-view (frame {cfg.frame})...")
        fig, axes = renderer.render_multi_view(scene, cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    return 0


def main_visualize(cfg: RuntimeConfig) -> int:
    """Visualize ground truth scene data."""
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        print_scene_info(scene)
        return 0

    err, cameras = validate_frame_and_cameras(scene, cfg)
    if err is not None:
        return err

    print(f"Using cameras: {cameras}")
    return render_scene(scene, cfg)


def main_predict_multiview(cfg: RuntimeConfig) -> int:
    """Run multi-view model predictions and visualize."""
    from src.plcs.inference.multiview_predictor import PLCSMultiViewPredictor

    if not _require_checkpoint(cfg):
        return 1

    print(f"Loading multi-view checkpoint from {cfg.checkpoint}...")
    predictor = PLCSMultiViewPredictor.load_from_checkpoint(
        cfg.checkpoint, device=cfg.device
    )

    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        print_scene_info(scene)
        return 0

    err, cameras = validate_frame_and_cameras(scene, cfg)
    if err is not None:
        return err

    num_views = len(cameras)
    num_frames = scene.meta["num_frames"]
    print(f"Running multi-view predictions using {num_views} cameras: {cameras}")
    print(f"Processing {num_frames} frames...")

    # Collect multi-view data for all frames
    human_kp_list = []
    court_kp_list = []
    human_vis_list = []
    court_vis_list = []

    for frame_idx in range(num_frames):
        frame_human_kp = []
        frame_court_kp = []
        frame_human_vis = []
        frame_court_vis = []

        for cam_idx in cameras:
            cam = scene.cameras[cam_idx]
            frame_human_kp.append(cam.human_kp_uv[frame_idx])
            frame_court_kp.append(cam.court_kp_uv[frame_idx])
            frame_human_vis.append(cam.human_kp_visible[frame_idx].astype(np.float32))
            frame_court_vis.append(cam.court_kp_visible[frame_idx].astype(np.float32))

        human_kp_list.append(np.stack(frame_human_kp, axis=0))
        court_kp_list.append(np.stack(frame_court_kp, axis=0))
        human_vis_list.append(np.stack(frame_human_vis, axis=0))
        court_vis_list.append(np.stack(frame_court_vis, axis=0))

    # Stack to (T, N, ...)
    human_kp_seq = torch.from_numpy(np.stack(human_kp_list, axis=0)).float()
    court_kp_seq = torch.from_numpy(np.stack(court_kp_list, axis=0)).float()
    human_vis_seq = torch.from_numpy(np.stack(human_vis_list, axis=0)).float()
    court_vis_seq = torch.from_numpy(np.stack(court_vis_list, axis=0)).float()

    # Run sequence prediction
    pred = predictor.predict_sequence(
        human_kp_seq=human_kp_seq,
        court_kp_seq=court_kp_seq,
        human_kp_mask_seq=human_vis_seq,
        court_kp_mask_seq=court_vis_seq,
        view_mask=None,
        denormalize=False,
    )

    # Overwrite SceneData with predictions
    scene.position[...] = pred["position"].numpy()
    scene.rotation[...] = pred["rotation"].numpy()

    return render_scene(scene, cfg)


@hydra.main(
    config_path="../configs", config_name="visualize_multiview", version_base="1.3"
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Hydra entry point for multi-view visualization and prediction."""
    runtime_cfg = build_runtime_config(cfg)

    if runtime_cfg.mode == "visualize":
        return main_visualize(runtime_cfg)
    if runtime_cfg.mode in {"predict", "predict-multiview", "predict_multiview"}:
        return main_predict_multiview(runtime_cfg)

    print(f"Unknown mode: {runtime_cfg.mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
