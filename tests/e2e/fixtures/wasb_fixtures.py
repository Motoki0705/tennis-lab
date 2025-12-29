"""WASB test fixtures for e2e tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch


def create_minimal_wasb_dataset(output_dir: Path | str) -> Path:
    """Create a minimal WASB dataset for testing.

    This function creates a dataset with the structure:
    - game1/Clip1/ with synthetic frames and Label.csv
    - game1/Clip2/ (similar)
    - meta.json

    Args:
        output_dir: Output directory for the dataset

    Returns:
        Path: Output directory containing the dataset

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create game1 directory
    game_dir = output_dir / "game1"
    game_dir.mkdir(exist_ok=True)

    # Create 2 clips with sufficient frames for trajectory windows (need >= sequence_length)
    for clip_idx in range(1, 3):
        clip_dir = game_dir / f"Clip{clip_idx}"
        clip_dir.mkdir(exist_ok=True)

        # Create 130 synthetic frames (black images with white circle for ball)
        # trajectory dataset requires sequence_length=64, step=8, so we need at least 64+ frames
        num_frames = 130
        for frame_idx in range(num_frames):
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Draw a white circle to simulate ball (moving in a parabolic path)
            t = frame_idx / num_frames
            center_x = int(640 + 400 * t)
            center_y = int(360 - 200 * np.sin(t * np.pi))
            if 0 <= center_x < 1280 and 0 <= center_y < 720:
                cv2.circle(img, (center_x, center_y), 10, (255, 255, 255), -1)

            # Save as .jpg (not .png) as expected by WASB
            frame_path = clip_dir / f"{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), img)

        # Create Label.csv with correct format: file name, visibility, x-coordinate, y-coordinate, status, score
        label_lines = ["file name,visibility,x-coordinate,y-coordinate,status,score"]
        for frame_idx in range(num_frames):
            t = frame_idx / num_frames
            center_x = int(640 + 400 * t)
            center_y = int(360 - 200 * np.sin(t * np.pi))
            visibility = 1 if 0 <= center_x < 1280 and 0 <= center_y < 720 else 0
            # Format: filename, visibility (0/1/2), x, y, status (0), score (0.0 for non-detections)
            label_lines.append(f"{frame_idx}.jpg,{visibility},{center_x},{center_y},0,0.0")

        label_path = clip_dir / "Label.csv"
        label_path.write_text("\n".join(label_lines) + "\n")

    return output_dir


def create_minimal_wasb_checkpoint(checkpoint_path: Path | str) -> Path:
    """Create a minimal WASB checkpoint for testing.

    This creates a minimal checkpoint compatible with WASB predictor loading.
    Note: This is a simplified checkpoint and may not work with actual WASB
    predictor, but is sufficient for script e2e tests.

    Args:
        checkpoint_path: Path where checkpoint will be saved

    Returns:
        Path: Checkpoint path

    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a minimal state dict
    # Include both 'state_dict' (Lightning format) and 'model_state_dict' (legacy format)
    checkpoint = {
        "state_dict": {},
        "model_state_dict": {},  # Legacy WASB format
        "hyper_parameters": {
            "model": "tracknet",
            "input_channels": 9,
            "output_channels": 3,
        },
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def create_minimal_trajectory_checkpoint(checkpoint_path: Path | str) -> Path:
    """Create a minimal trajectory completion checkpoint for testing.

    This creates a checkpoint compatible with TrajectoryLightningModule using
    TrajectoryBiLSTM model architecture.

    Args:
        checkpoint_path: Path where checkpoint will be saved

    Returns:
        Path: Checkpoint path

    """
    from src.wasb.models.trajectory_completion import TrajectoryBiLSTM

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create actual TrajectoryBiLSTM model to get valid state_dict
    model = TrajectoryBiLSTM(hidden_dim=64, num_layers=2, dropout=0.1)

    # Build state_dict with "model." prefix as used by TrajectoryLightningModule
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}

    checkpoint = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "config": {
                "model": {
                    "name": "trajectory_bilstm",
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                },
                "training": {
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                },
            },
        },
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def create_minimal_video(
    video_path: Path | str,
    num_frames: int = 100,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Path:
    """Create a minimal synthetic video for testing.

    The video contains black frames with a moving white circle to simulate a ball.

    Args:
        video_path: Path where video will be saved
        num_frames: Number of frames to generate (default: 100)
        width: Video width (default: 1280)
        height: Video height (default: 720)
        fps: Frames per second (default: 30)

    Returns:
        Path: Video path

    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
    )

    # Generate frames with moving ball
    for frame_idx in range(num_frames):
        # Create black frame
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw white circle (ball) moving in a parabolic path
        t = frame_idx / num_frames
        center_x = int(width * 0.2 + width * 0.6 * t)
        center_y = int(height * 0.7 - height * 0.4 * np.sin(t * np.pi))

        cv2.circle(img, (center_x, center_y), 10, (255, 255, 255), -1)

        # Write frame
        out.write(img)

    out.release()

    return video_path
