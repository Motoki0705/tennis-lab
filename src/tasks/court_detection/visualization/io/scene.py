"""Scene IO helpers for court detection visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image

from src.court_detection.visualization.types import RuntimeConfig, SceneImage


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (int(raw[0]), int(raw[1]), int(raw[2]))
    return default


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.get("run", {}) or {}

    run_device = str(run.get("device", vis.get("device", "auto")))
    output_dir = vis.get("output_dir", run.get("output_dir", "outputs/court_detection/visualize"))

    checkpoint_raw = vis.get("checkpoint")
    checkpoint = Path(to_absolute_path(str(checkpoint_raw))) if checkpoint_raw else None

    return RuntimeConfig(
        mode=str(vis.get("mode", "visualize")),
        input_path=Path(to_absolute_path(str(vis.input_path))),
        output_dir=Path(to_absolute_path(str(output_dir))),
        num_samples=max(1, int(vis.get("num_samples", 10))),
        checkpoint=checkpoint,
        device=_resolve_device(run_device),
        save_overlay=bool(vis.get("save_overlay", True)),
        save_json=bool(vis.get("save_json", False)),
        point_radius=max(1, int(vis.get("point_radius", 5))),
        point_color=_parse_color(vis.get("point_color"), (0, 255, 0)),
        line_color=_parse_color(vis.get("line_color"), (255, 255, 0)),
        text_color=_parse_color(vis.get("text_color"), (255, 255, 255)),
        line_thickness=max(1, int(vis.get("line_thickness", 2))),
        show_keypoint_ids=bool(vis.get("show_keypoint_ids", True)),
        show_court_lines=bool(vis.get("show_court_lines", True)),
        visibility_threshold=float(vis.get("visibility_threshold", 0.5)),
        hydra_cfg=cfg,
    )


def collect_input_paths(input_path: Path, num_samples: int) -> list[Path]:
    """Collect input image files from a file path or directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    image_files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_files.extend(input_path.glob(pattern))
    image_files = sorted(image_files)
    return image_files[:num_samples]


def load_scene_images(cfg: RuntimeConfig) -> list[SceneImage]:
    """Load scene images as RGB numpy arrays."""
    image_paths = collect_input_paths(cfg.input_path, cfg.num_samples)
    scenes: list[SceneImage] = []
    for image_path in image_paths:
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        scenes.append(SceneImage(image_path=image_path, image_rgb=image_rgb))
    return scenes
