"""Visualize BLCS scenes and optionally run model predictions (Hydra-based).

Example commands:
    `uv run python -m src.blcs.scripts.visualize`
    `uv run python -m src.blcs.scripts.visualize visualization.scene_path=data/blcs/scenes/scene_000000.npz visualization.info=true`
    `uv run python -m src.blcs.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/blcs/checkpoints/last.ckpt`

Config entry point: `src/blcs/configs/visualize.yaml`
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.blcs.generate_dataset.io.dataset_io import load_scene
from src.utils.rendering import BLCSSceneRenderer

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime arguments for visualization/prediction."""

    mode: str
    scene_path: Path
    frame: int
    view: str
    camera: int
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

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        frame=int(vis.frame),
        view=str(vis.view),
        camera=int(vis.camera),
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        save_input=Path(to_absolute_path(str(vis.save_input))) if vis.save_input else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        device=_resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
    )


def validate_frame_and_camera(scene: dict[str, Any], cfg: RuntimeConfig) -> int | None:
    """Validate (frame, camera) indices against the loaded scene."""
    num_frames = int(scene["ball_pos_world"].shape[0])
    if cfg.frame < 0 or cfg.frame >= num_frames:
        print(f"Error: Frame {cfg.frame} out of range (0-{num_frames - 1})")
        return 1

    num_cameras = int(scene["num_cameras"])
    if cfg.camera < 0 or cfg.camera >= num_cameras:
        print(f"Error: Camera {cfg.camera} out of range (0-{num_cameras - 1})")
        return 1

    return None


def save_input_scene(scene: dict[str, Any], cfg: RuntimeConfig) -> None:
    """Save 2D input scene animation (camera view)."""
    if cfg.save_input is None:
        return

    renderer = BLCSSceneRenderer()
    meta = scene["meta"]
    fps = cfg.fps or float(meta.get("fps_out", 30.0))

    print(f"Creating 2D input scene animation (camera view)...")
    anim = renderer.create_animation(
        scene,
        view="camera",
        camera_idx=cfg.camera,
        fps=fps,
    )

    if anim is None:
        print("Error: Failed to create input scene animation")
        return

    cfg.save_input.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving input scene animation to {cfg.save_input}...")
    anim.save(str(cfg.save_input), fps=fps)
    plt.close()
    print("Done!")


def render_scene(scene: dict[str, Any], cfg: RuntimeConfig) -> int:
    """Render a scene with the configured view settings."""
    renderer = BLCSSceneRenderer()

    if cfg.view == "animation":
        meta = scene["meta"]
        fps = cfg.fps or float(meta.get("fps_out", 30.0))
        print(f"Creating animation ({cfg.animation_view} view)...")
        anim = renderer.create_animation(
            scene,
            view=cfg.animation_view,
            camera_idx=cfg.camera,
            fps=fps,
        )

        if anim is None:
            return 1

        if cfg.save:
            print(f"Saving animation to {cfg.save}...")
            anim.save(str(cfg.save), fps=fps)
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

    elif cfg.view == "camera":
        print(f"Rendering camera {cfg.camera} view (frame {cfg.frame})...")
        fig, _ax = renderer.render_camera_view(
            scene, camera_idx=cfg.camera, frame_idx=cfg.frame
        )

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

    err = validate_frame_and_camera(scene, cfg)
    if err is not None:
        return err

    save_input_scene(scene, cfg)
    return render_scene(scene, cfg)


def main_predict(cfg: RuntimeConfig) -> int:
    """Run the BLCS predictor and visualize its trajectory output."""
    if cfg.checkpoint is None:
        print("Error: visualization.checkpoint must be set for predict mode.")
        return 1

    from src.blcs.inference.predictor import BLCSPredictor

    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint, device=cfg.device
    )

    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    renderer = BLCSSceneRenderer()

    if cfg.info:
        renderer.print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, cfg)
    if err is not None:
        return err

    cam = scene["cameras"][cfg.camera]
    ball_uv = cam["ball_uv"]
    ball_vis = cam["ball_visible"]
    court_kp = cam["court_kp_uv"]
    court_vis = cam["court_kp_visible"]

    ball_uv_t = torch.from_numpy(ball_uv).float()
    court_kp_t = torch.from_numpy(court_kp).float()
    ball_mask_t = torch.from_numpy(ball_vis.astype(np.float32))
    court_vis_t = torch.from_numpy(court_vis.astype(np.float32))

    print("Running BLCS prediction...")
    outputs = predictor.predict(
        ball_uv=ball_uv_t,
        court_kp=court_kp_t,
        ball_mask=ball_mask_t,
        court_vis=court_vis_t,
        denormalize=True,
    )

    if cfg.output is not None:
        output_path = Path(cfg.output)
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

    save_input_scene(scene, cfg)

    gt_pos = scene["ball_pos_world"]
    pred_pos = outputs["position"].squeeze(0).cpu().numpy()

    if cfg.view == "animation" and cfg.animation_view in ("3d", "2d"):
        meta = scene["meta"]
        fps = cfg.fps or float(meta.get("fps_out", 30.0))
        print(f"Creating comparison animation ({cfg.animation_view} view)...")
        anim = renderer.create_comparison_animation(
            gt_positions=gt_pos,
            pred_positions=pred_pos,
            view=cfg.animation_view,
            fps=fps,
            title="GT vs Prediction",
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

    scene_pred = dict(scene)
    scene_pred["ball_pos_world"] = pred_pos

    return render_scene(scene_pred, cfg)


@hydra.main(config_path="../configs", config_name="visualize", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    runtime = build_runtime_config(cfg)
    if runtime.mode == "visualize":
        return main_visualize(runtime)
    if runtime.mode == "predict":
        return main_predict(runtime)
    print(f"Error: unknown visualization.mode '{runtime.mode}' (expected visualize|predict)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
