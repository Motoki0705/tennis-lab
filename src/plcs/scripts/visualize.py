#!/usr/bin/env python
"""Unified CLI for PLCS scene visualization and prediction.

This script provides three modes:
- visualize: Visualize ground truth scene data
- predict: Run single-frame model predictions and visualize
- predict-seq: Run sequence model predictions and visualize

Usage:
    # Ground truth visualization
    python -m plcs.scripts.visualize visualize data/plcs_scenes/scene_000000.npz --view 3d

    # Single-frame model prediction
    python -m plcs.scripts.visualize predict data/plcs_scenes/scene_000000.npz \\
        --checkpoint outputs/plcs/checkpoints/last.ckpt --view animation

    # Sequence model prediction
    python -m plcs.scripts.visualize predict-seq data/plcs_scenes/scene_000000.npz \\
        --checkpoint outputs/plcs_sequence/checkpoints/last.ckpt --view multi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.plcs.data.scene_generator import SceneGenerator
from src.rendering import PLCSSceneRenderer as SceneRenderer

if TYPE_CHECKING:
    from src.plcs.data.scene_generator import SceneData


# =============================================================================
# Common Utilities
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all subcommands."""
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Path to scene NPZ file",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize (for static views)",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["3d", "2d", "camera", "multi", "animation"],
        default="multi",
        help="View type",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for camera view and prediction",
    )
    parser.add_argument(
        "--animation-view",
        type=str,
        choices=["3d", "2d_topdown", "camera"],
        default="2d_topdown",
        help="View type for animation",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for animation (default: use scene FPS)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure/animation to file",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print scene info and exit",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related arguments for predict subcommands."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )


def print_scene_info(scene: SceneData) -> None:
    """Print scene metadata and statistics."""
    meta = scene.meta
    print("=" * 60)
    print("Scene Information")
    print("=" * 60)
    print(f"Scene ID:        {meta['scene_id']}")
    print(f"Motion source:   {meta['motion_source']}")
    print(f"Category:        {meta['motion_category']}")
    print(f"Gender:          {meta['gender']}")
    print(f"FPS:             {meta['fps']}")
    print(f"Num frames:      {meta['num_frames']}")
    print(f"Duration:        {meta['num_frames'] / meta['fps']:.2f} seconds")
    print(
        f"Initial pos:     ({meta['initial_position'][0]:.2f}, {meta['initial_position'][1]:.2f})"
    )
    print(f"Initial yaw:     {np.degrees(meta['initial_yaw']):.1f}°")
    print(f"Cameras sampled: {meta['num_cameras_sampled']}")
    print(f"Cameras kept:    {meta['num_cameras_filtered']}")
    print()
    print("Position statistics (normalized):")
    print(
        f"  X range: [{scene.position[:, 0].min():.3f}, {scene.position[:, 0].max():.3f}]"
    )
    print(
        f"  Y range: [{scene.position[:, 1].min():.3f}, {scene.position[:, 1].max():.3f}]"
    )
    print(
        f"  Z range: [{scene.position[:, 2].min():.3f}, {scene.position[:, 2].max():.3f}]"
    )
    print()
    print("Camera visibility:")
    for i, cam in enumerate(scene.cameras):
        print(
            f"  Camera {i}: Human {cam.human_visibility_ratio:.1%}, Court {cam.court_visibility_count:.1f}/20"
        )


def validate_frame_and_camera(scene: SceneData, args: argparse.Namespace) -> int | None:
    """Validate frame and camera indices.

    Returns:
        None if validation passes, error code (int) otherwise.

    """
    num_frames = scene.meta["num_frames"]
    if args.frame >= num_frames:
        print(f"Error: Frame {args.frame} out of range (0-{num_frames - 1})")
        return 1

    num_cameras = len(scene.cameras)
    if args.camera >= num_cameras:
        print(f"Error: Camera {args.camera} out of range (0-{num_cameras - 1})")
        return 1

    return None


def render_scene(scene: SceneData, args: argparse.Namespace) -> int:
    """Render scene based on view type.

    Returns:
        Exit code (0 for success).

    """
    renderer = SceneRenderer()

    if args.view == "animation":
        print(f"Creating animation ({args.animation_view} view)...")
        anim = renderer.create_animation(
            scene,
            view=args.animation_view,
            camera_idx=args.camera,
            fps=args.fps,
        )

        if args.save:
            print(f"Saving animation to {args.save}...")
            anim.save(str(args.save), fps=args.fps or scene.meta["fps"])
            print("Done!")
        else:
            plt.show()

    elif args.view == "3d":
        print(f"Rendering 3D view (frame {args.frame})...")
        fig, ax = renderer.render_frame_3d(scene, args.frame)

        if args.save:
            fig.savefig(str(args.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()

    elif args.view == "2d":
        print(f"Rendering 2D top-down view (frame {args.frame})...")
        fig, ax = renderer.render_frame_2d_topdown(scene, args.frame)

        if args.save:
            fig.savefig(str(args.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()

    elif args.view == "camera":
        print(f"Rendering camera {args.camera} view (frame {args.frame})...")
        fig, ax = renderer.render_camera_view(scene, args.frame, args.camera)

        if args.save:
            fig.savefig(str(args.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()

    elif args.view == "multi":
        print(f"Rendering multi-view (frame {args.frame})...")
        fig, axes = renderer.render_multi_view(scene, args.frame)

        if args.save:
            fig.savefig(str(args.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()

    return 0


# =============================================================================
# Subcommand: visualize (Ground Truth)
# =============================================================================


def main_visualize(args: argparse.Namespace) -> int:
    """Visualize ground truth scene data."""
    print(f"Loading scene from {args.scene_path}...")
    scene = SceneGenerator.load_scene(args.scene_path)

    if args.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, args)
    if err is not None:
        return err

    return render_scene(scene, args)


# =============================================================================
# Subcommand: predict (Single-Frame Model)
# =============================================================================


def main_predict(args: argparse.Namespace) -> int:
    """Run single-frame model predictions and visualize."""
    from src.plcs.inference.predictor import PLCSPredictor

    print(f"Loading checkpoint from {args.checkpoint}...")
    predictor = PLCSPredictor.load_from_checkpoint(args.checkpoint, device=args.device)

    print(f"Loading scene from {args.scene_path}...")
    scene = SceneGenerator.load_scene(args.scene_path)

    if args.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, args)
    if err is not None:
        return err

    # Run predictions and overwrite SceneData
    num_frames = scene.meta["num_frames"]
    cam = scene.cameras[args.camera]

    print(f"Running predictions for {num_frames} frames using camera {args.camera}...")
    for frame_idx in range(num_frames):
        human_kp = cam.human_kp_uv[frame_idx]
        court_kp = cam.court_kp_uv[frame_idx]
        human_vis = cam.human_kp_visible[frame_idx]
        court_vis = cam.court_kp_visible[frame_idx]

        pred = predictor.predict(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )

        # Overwrite normalized position and rotation
        scene.position[frame_idx] = pred["position"].numpy()
        scene.rotation[frame_idx] = pred["rotation"].numpy()

    return render_scene(scene, args)


# =============================================================================
# Subcommand: predict-seq (Sequence Model)
# =============================================================================


def main_predict_sequence(args: argparse.Namespace) -> int:
    """Run sequence model predictions and visualize."""
    from src.plcs.inference.sequence_predictor import PLCSSequencePredictor

    print(f"Loading sequence checkpoint from {args.checkpoint}...")
    predictor = PLCSSequencePredictor.load_from_checkpoint(
        args.checkpoint, device=args.device
    )

    print(f"Loading scene from {args.scene_path}...")
    scene = SceneGenerator.load_scene(args.scene_path)

    if args.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, args)
    if err is not None:
        return err

    # Run sequence prediction and overwrite SceneData
    cam = scene.cameras[args.camera]

    # Prepare sequence inputs: (T, K, 2) and (T, K)
    human_kp_seq = torch.from_numpy(cam.human_kp_uv).float()  # (T, 17, 2)
    court_kp_seq = torch.from_numpy(cam.court_kp_uv).float()  # (T, 20, 2)
    human_vis_seq = torch.from_numpy(cam.human_kp_visible).float()  # (T, 17)
    court_vis_seq = torch.from_numpy(cam.court_kp_visible).float()  # (T, 20)

    num_frames = scene.meta["num_frames"]
    print(
        f"Running sequence prediction for {num_frames} frames using camera {args.camera}..."
    )

    pred = predictor.predict(
        human_kp=human_kp_seq,
        court_kp=court_kp_seq,
        human_vis=human_vis_seq,
        court_vis=court_vis_seq,
        denormalize=False,  # Keep normalized for SceneData
    )

    # Overwrite SceneData with predictions
    scene.position[...] = pred["position"].cpu().numpy()  # (T, 3)
    scene.rotation[...] = pred["rotation"].cpu().numpy()  # (T, 2)

    return render_scene(scene, args)


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="PLCS scene visualization and prediction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Subcommand: visualize
    parser_vis = subparsers.add_parser(
        "visualize",
        help="Visualize ground truth scene data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser_vis)

    # Subcommand: predict
    parser_pred = subparsers.add_parser(
        "predict",
        help="Run single-frame model predictions and visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser_pred)
    add_model_args(parser_pred)

    # Subcommand: predict-seq
    parser_seq = subparsers.add_parser(
        "predict-seq",
        help="Run sequence model predictions and visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser_seq)
    add_model_args(parser_seq)

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.command == "visualize":
        return main_visualize(args)
    elif args.command == "predict":
        return main_predict(args)
    elif args.command == "predict-seq":
        return main_predict_sequence(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
