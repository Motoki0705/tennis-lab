"""Visualization helpers for event detection.

This module contains small, typed utilities shared by the UV and 3D
visualization CLIs.

Notes:
- We keep the label generation logic consistent with
  `src.event_detection.data.dataset.BLCSRallyEventDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.common.data.npz_meta import decode_meta


@dataclass(frozen=True)
class EventLabelConfig:
    """Configuration for event label generation (shot/bounce)."""

    sigma_frames: float = 2.5
    shot_time_key: str = "t_start"
    bounce_time_key: str = "t_bounce1"


def select_camera(camera: Any, num_cameras: int) -> int:
    """Select a camera index from config values."""

    if num_cameras <= 0:
        return 0
    if camera is None:
        return 0
    if camera == "random":
        return int(torch.randint(0, num_cameras, (1,)).item())
    if isinstance(camera, int):
        return min(max(int(camera), 0), num_cameras - 1)
    if isinstance(camera, str) and camera.isdigit():
        return min(max(int(camera), 0), num_cameras - 1)
    return 0


def extract_event_indices(
    meta: dict[str, Any], *, cfg: EventLabelConfig
) -> tuple[list[int], list[int]]:
    """Extract shot/bounce frame indices from rally metadata."""

    shots = meta.get("shots", []) or []
    shot_times: list[int] = []
    bounce_times: list[int] = []

    for s in shots:
        if not isinstance(s, dict):
            continue
        t_shot = int(s.get(cfg.shot_time_key, -1))
        t_bounce = int(s.get(cfg.bounce_time_key, -1))
        if t_shot >= 0:
            shot_times.append(t_shot)
        if t_bounce >= 0:
            bounce_times.append(t_bounce)

    shot_times = sorted(set(shot_times))
    bounce_times = sorted(set(bounce_times))
    return shot_times, bounce_times


def gaussian_soft_labels(
    length: int,
    event_indices: list[int],
    *,
    sigma: float,
    device: torch.device,
) -> Tensor:
    """Create soft labels with Gaussian peaks at the given frame indices."""

    if length <= 0:
        return torch.zeros((0,), device=device)
    if not event_indices:
        return torch.zeros((length,), device=device)

    t = torch.arange(length, device=device, dtype=torch.float32)
    out = torch.zeros((length,), device=device, dtype=torch.float32)
    denom = 2.0 * float(sigma) * float(sigma)

    for idx in event_indices:
        if 0 <= idx < length:
            out = torch.maximum(out, torch.exp(-((t - float(idx)) ** 2) / denom))

    return out


def build_targets(
    length: int,
    *,
    shot_indices: list[int],
    bounce_indices: list[int],
    cfg: EventLabelConfig,
    device: torch.device,
) -> Tensor:
    """Build stacked targets of shape (T, 2) for (shot, bounce)."""

    y_shot = gaussian_soft_labels(
        length,
        shot_indices,
        sigma=float(cfg.sigma_frames),
        device=device,
    )
    y_bounce = gaussian_soft_labels(
        length,
        bounce_indices,
        sigma=float(cfg.sigma_frames),
        device=device,
    )
    return torch.stack([y_shot, y_bounce], dim=-1)


def save_outputs(outputs: dict[str, Any], output_path: Path) -> None:
    """Save prediction outputs as .pt or .json."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".pt":
        torch.save(outputs, output_path)
        return

    if output_path.suffix == ".json":
        json_data: dict[str, Any] = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                json_data[k] = v.squeeze(0).cpu().tolist()
            else:
                json_data[k] = v
        output_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    raise ValueError(
        f"Unsupported output format: {output_path.suffix} (expected .pt or .json)"
    )
