"""Orchestrate PLCS scene visualization and prediction animation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.plcs.generate_dataset.io.dataset_io import load_scene
from src.plcs.inference.predictor import PLCSPredictor
from src.plcs.visualization.rendering import PLCSSceneRenderer


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for PLCS visualization."""

    mode: str
    scene_path: Path
    checkpoint: Path | None
    device: str
    animation_view: str
    fps: float | None
    save: Path | None
    camera: int
    cameras: list[int] | str | None
    info: bool


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _parse_cameras(raw_value: object) -> list[int] | str | None:
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return None
        if stripped == "all":
            return "all"
        return [int(part.strip()) for part in stripped.split(",")]
    return [int(v) for v in raw_value]


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.get("run", {})
    run_device = run.get("device", vis.get("device", "auto"))

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        checkpoint=Path(to_absolute_path(str(vis.checkpoint))) if vis.checkpoint else None,
        device=_resolve_device(str(run_device)),
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        camera=int(vis.get("camera", 0)),
        cameras=_parse_cameras(vis.get("cameras")),
        info=bool(vis.info),
    )


def _resolve_cameras(cfg: RuntimeConfig, num_cameras: int) -> list[int]:
    if cfg.cameras == "all":
        return list(range(num_cameras))
    if isinstance(cfg.cameras, list):
        return cfg.cameras
    return [cfg.camera]


def _print_scene_info(scene: object) -> None:
    meta = getattr(scene, "meta", {})
    print("=" * 60)
    print("Scene Information")
    print("=" * 60)
    print(f"Scene ID:        {meta.get('scene_id', 'unknown')}")
    print(f"Motion source:   {meta.get('motion_source', 'unknown')}")
    print(f"Category:        {meta.get('motion_category', 'unknown')}")
    print(f"FPS:             {meta.get('fps', '?')}")
    print(f"Num frames:      {meta.get('num_frames', '?')}")
    print(f"Num cameras:     {len(getattr(scene, 'cameras', []))}")


def _predict_full_sequence(
    predictor: PLCSPredictor,
    scene: object,
    cameras: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    human_kp = np.stack([scene.cameras[c].human_kp_uv for c in cameras], axis=0)
    court_kp = np.stack([scene.cameras[c].court_kp_uv for c in cameras], axis=0)
    human_vis = np.stack([scene.cameras[c].human_kp_visible.astype(np.float32) for c in cameras], axis=0)
    court_vis = np.stack([scene.cameras[c].court_kp_visible.astype(np.float32) for c in cameras], axis=0)

    human_kp_t = torch.from_numpy(human_kp).float()
    court_kp_t = torch.from_numpy(court_kp).float()
    human_vis_t = torch.from_numpy(human_vis).float()
    court_vis_t = torch.from_numpy(court_vis).float()
    human_mask_t = torch.ones(human_kp_t.shape[0], human_kp_t.shape[1], dtype=torch.float32)

    pred = predictor.predict(
        human_kp=human_kp_t,
        court_kp=court_kp_t,
        human_vis=human_vis_t,
        human_mask=human_mask_t,
        court_vis=court_vis_t,
        denormalize=False,
    )
    position = pred["position"]
    rotation = pred["rotation"]
    if position.dim() == 2:
        position = position.unsqueeze(1)
    if rotation.dim() == 2:
        rotation = rotation.unsqueeze(1)

    return position.squeeze(0).numpy(), rotation.squeeze(0).numpy()


def _predict_frame_by_frame(
    predictor: PLCSPredictor,
    scene: object,
    cameras: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(scene.meta["num_frames"])
    pos_list: list[np.ndarray] = []
    rot_list: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        human_kp = np.stack([scene.cameras[c].human_kp_uv[frame_idx : frame_idx + 1] for c in cameras], axis=0)
        court_kp = np.stack([scene.cameras[c].court_kp_uv[frame_idx : frame_idx + 1] for c in cameras], axis=0)
        human_vis = np.stack(
            [scene.cameras[c].human_kp_visible[frame_idx : frame_idx + 1].astype(np.float32) for c in cameras],
            axis=0,
        )
        court_vis = np.stack(
            [scene.cameras[c].court_kp_visible[frame_idx : frame_idx + 1].astype(np.float32) for c in cameras],
            axis=0,
        )

        pred = predictor.predict(
            human_kp=torch.from_numpy(human_kp).float(),
            court_kp=torch.from_numpy(court_kp).float(),
            human_vis=torch.from_numpy(human_vis).float(),
            human_mask=torch.ones(len(cameras), 1, dtype=torch.float32),
            court_vis=torch.from_numpy(court_vis).float(),
            denormalize=False,
        )
        pos = pred["position"]
        rot = pred["rotation"]
        if pos.dim() == 3:
            pos = pos[:, 0]
        if rot.dim() == 3:
            rot = rot[:, 0]
        pos_list.append(pos.squeeze(0).numpy())
        rot_list.append(rot.squeeze(0).numpy())

    return np.stack(pos_list, axis=0), np.stack(rot_list, axis=0)


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run PLCS visualization orchestration."""
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        _print_scene_info(scene)
        return 0

    num_cameras = len(scene.cameras)
    cameras = _resolve_cameras(cfg, num_cameras)
    for camera_idx in cameras:
        if camera_idx < 0 or camera_idx >= num_cameras:
            print(f"Error: camera index {camera_idx} is out of range [0, {num_cameras - 1}].")
            return 1

    if cfg.animation_view not in {"3d", "2d_topdown", "camera"}:
        print("Error: visualization.animation_view must be one of '3d', '2d_topdown', 'camera'.")
        return 1

    render_scene = scene
    mode = cfg.mode.strip().lower()
    if mode == "predict":
        if cfg.checkpoint is None:
            print("Error: visualization.checkpoint must be set for predict mode.")
            return 1
        predictor = PLCSPredictor.load_from_checkpoint(cfg.checkpoint, device=cfg.device)

        render_scene = copy.deepcopy(scene)
        try:
            pred_pos, pred_rot = _predict_full_sequence(predictor, scene, cameras)
        except ValueError as exc:
            if "supports T=1 only" not in str(exc):
                raise
            pred_pos, pred_rot = _predict_frame_by_frame(predictor, scene, cameras)

        render_scene.position[...] = pred_pos
        render_scene.rotation[...] = pred_rot
    elif mode != "visualize":
        print(f"Error: unknown visualization.mode '{cfg.mode}'.")
        return 1

    renderer = PLCSSceneRenderer()
    fps = cfg.fps or float(render_scene.meta.get("fps", 30.0))
    camera_idx = cameras[0] if cameras else cfg.camera
    anim = renderer.create_animation(
        render_scene,
        view=cfg.animation_view,
        camera_idx=camera_idx,
        fps=fps,
    )

    if cfg.save is not None:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(cfg.save), fps=fps)
        plt.close()
        print(f"Saved animation to {cfg.save}")
    else:
        plt.show()

    return 0
