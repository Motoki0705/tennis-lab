"""Visualize BLCS multi-view scenes and run model predictions (Hydra-based).

Example commands:
    `uv run python -m src.blcs.scripts.visualize_multiview`
    `uv run python -m src.blcs.scripts.visualize_multiview \
        visualization.scene_path=data/blcs/scenes/scene_000000.npz \
        visualization.mode=predict \
        visualization.checkpoint=outputs/blcs_multiview/checkpoints/last.ckpt`

Config entry point: `src/blcs/configs/visualize_multiview.yaml`
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.blcs.generate_dataset.io.dataset_io import load_scene
from src.utils.rendering import BLCSSceneRenderer


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime arguments for visualization/prediction."""

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
    output: str | None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from the composed Hydra config."""
    vis = cfg.visualization
    run = cfg.run

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
        device=_resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
    )


def _require_checkpoint(cfg: RuntimeConfig) -> bool:
    if cfg.checkpoint is None:
        print("Error: checkpoint must be provided for prediction modes.")
        return False
    return True


def validate_frame_and_cameras(
    scene: dict[str, Any], cfg: RuntimeConfig
) -> tuple[int | None, list[int]]:
    """Validate (frame, cameras) indices against the loaded scene."""
    num_frames = int(scene["ball_pos_world"].shape[0])
    if cfg.frame < 0 or cfg.frame >= num_frames:
        print(f"Error: Frame {cfg.frame} out of range (0-{num_frames - 1})")
        return 1, []

    num_cameras = int(scene["num_cameras"])
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


def render_scene(scene: dict[str, Any], cfg: RuntimeConfig) -> int:
    """Render a scene with the configured view settings."""
    renderer = BLCSSceneRenderer()

    if cfg.view == "animation":
        meta = scene["meta"]
        fps = cfg.fps or float(meta.get("fps_out", 30.0))
        camera_idx = cfg.cameras[0] if cfg.cameras else 0
        print(f"Creating animation ({cfg.animation_view} view)...")
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
        fig, _ax = renderer.render_3d_view(scene, frame_idx=cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "2d":
        print(f"Rendering 2D top-down view (frame {cfg.frame})...")
        fig, _ax = renderer.render_2d_topdown(scene, frame_idx=cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "multi":
        print(f"Rendering multi-view (frame {cfg.frame})...")
        fig = renderer.render_multi_view(scene, frame_idx=cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    else:
        print(f"Error: unknown view '{cfg.view}'")
        return 1

    return 0


def main_visualize(cfg: RuntimeConfig) -> int:
    """Visualize ground-truth scenes."""
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    renderer = BLCSSceneRenderer()

    if cfg.info:
        renderer.print_scene_info(scene)
        return 0

    err, cameras = validate_frame_and_cameras(scene, cfg)
    if err is not None:
        return err

    print(f"Using cameras: {cameras}")
    return render_scene(scene, cfg)


def main_predict_multiview(cfg: RuntimeConfig) -> int:
    """Run multi-view model predictions and visualize."""
    from src.blcs.inference.multiview_predictor import BLCSMultiViewPredictor

    if not _require_checkpoint(cfg):
        return 1

    checkpoint = cfg.checkpoint
    if checkpoint is None:
        print("Error: checkpoint is required for prediction.")
        return 1

    print(f"Loading multi-view checkpoint from {checkpoint}...")
    predictor = BLCSMultiViewPredictor.load_from_checkpoint(
        checkpoint, device=cfg.device
    )

    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    renderer = BLCSSceneRenderer()

    if cfg.info:
        renderer.print_scene_info(scene)
        return 0

    err, cameras = validate_frame_and_cameras(scene, cfg)
    if err is not None:
        return err

    num_views = len(cameras)
    num_frames = int(scene["ball_pos_world"].shape[0])
    print(f"Running multi-view predictions using {num_views} cameras: {cameras}")
    print(f"Processing {num_frames} frames...")

    # Collect multi-view data
    ball_uv_list = []
    court_kp_list = []
    ball_vis_list = []
    court_vis_list = []

    for cam_idx in cameras:
        cam = scene["cameras"][cam_idx]
        ball_uv_list.append(cam["ball_uv"])  # (T, 2)
        court_kp_list.append(cam["court_kp_uv"])  # (20, 2)
        ball_vis_list.append(cam["ball_visible"].astype(np.float32))  # (T,)
        court_vis_list.append(cam["court_kp_visible"].astype(np.float32))  # (20,)

    # Stack to (N, T, 2) and (N, 20, 2)
    ball_uv = torch.from_numpy(np.stack(ball_uv_list, axis=0)).float()  # (N, T, 2)
    court_kp = torch.from_numpy(np.stack(court_kp_list, axis=0)).float()  # (N, 20, 2)
    ball_mask = torch.from_numpy(np.stack(ball_vis_list, axis=0)).float()  # (N, T)
    court_vis = torch.from_numpy(np.stack(court_vis_list, axis=0)).float()  # (N, 20)

    # Run prediction
    outputs = predictor.predict(
        ball_uv=ball_uv,
        court_kp=court_kp,
        ball_mask=ball_mask,
        court_vis=court_vis,
        num_views=None,
        denormalize=True,
    )

    # Save output if requested
    if cfg.output is not None:
        output_path = Path(cfg.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".pt":
            torch.save(outputs, output_path)
        elif output_path.suffix == ".json":
            json_data = {k: v.squeeze(0).cpu().tolist() for k, v in outputs.items()}
            output_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            print(
                f"Warning: Unknown output format '{output_path.suffix}', "
                "only .pt and .json are supported. Skipping save."
            )
        print(f"Saved prediction outputs to {output_path}")

    # Get predictions
    pred_pos = outputs["position"].squeeze(0).cpu().numpy()  # (T, 3)
    if "position_meters" in outputs:
        pred_pos = outputs["position_meters"].squeeze(0).cpu().numpy()

    gt_pos = scene["ball_pos_world"]

    # Create comparison visualization if animation view
    if cfg.view == "animation" and cfg.animation_view in ("3d", "2d"):
        meta = scene["meta"]
        fps = cfg.fps or float(meta.get("fps_out", 30.0))
        print(f"Creating comparison animation ({cfg.animation_view} view)...")
        anim = renderer.create_comparison_animation(
            gt_positions=gt_pos,
            pred_positions=pred_pos,
            view=cfg.animation_view,
            fps=fps,
            title="GT vs Multi-View Prediction",
        )

        if anim is None:
            return 1

        if cfg.save:
            cfg.save.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving comparison animation to {cfg.save}...")
            anim.save(str(cfg.save), fps=fps)
            plt.close()
            print("Done!")
        else:
            plt.show()
        return 0

    # Replace scene trajectory with prediction for other views
    scene_pred = dict(scene)
    scene_pred["ball_pos_world"] = pred_pos

    return render_scene(scene_pred, cfg)


TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def hydra_main(**kwargs: Any) -> Callable[[TFunc], TFunc]:
    """Typed wrapper around hydra.main for mypy."""
    return cast(Callable[[TFunc], TFunc], hydra.main(**kwargs))


@hydra_main(
    config_path="../configs", config_name="visualize_multiview", version_base="1.3"
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point for multi-view visualization."""
    runtime = build_runtime_config(cfg)
    if runtime.mode == "visualize":
        return main_visualize(runtime)
    if runtime.mode in {"predict", "predict-multiview", "predict_multiview"}:
        return main_predict_multiview(runtime)
    print(
        f"Error: unknown visualization.mode '{runtime.mode}' "
        "(expected visualize|predict)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
