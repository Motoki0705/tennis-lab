from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.ball_detection.models.mdd_pretrain import DeepMDDQueryDetector
from src.tasks.ball_detection.training.coordinate_compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import CoordinateMetrics
from src.tasks.ball_detection.training.coordinate_images import coordinate_batches
from src.tasks.ball_detection.training.heatmap_pretraining.objective import image_input

from .augmentation import VideoAugmenter


def save_query_preview(path: Path, rgb: Tensor, prediction: Tensor, batch: dict[str, Any],
                       occluded: Tensor | None = None) -> None:
    pictures = []
    images = rgb[0].detach().cpu()
    for t in range(32):
        frame = Image.fromarray(images[t].permute(1, 2, 0).numpy()).resize((640, 360))
        canvas = Image.new("RGB", (640, 384), "white")
        canvas.paste(frame, (0, 24))
        draw = ImageDraw.Draw(canvas)
        for point, valid, color in ((batch["uv"][0, t], bool(batch["position_valid"][0, t]), "lime"),
                                    (prediction[0, t], True, "orange")):
            if valid:
                x, y = float(point[0]) * 639, 24 + float(point[1]) * 359
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        hidden = bool(occluded[0, t]) if occluded is not None else False
        draw.text((4, 4), f"frame {int(batch['frame_indices'][0,t])} | GT green / prediction orange | artificial occlusion={hidden}", fill="black")
        pictures.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=100, loop=0)


def evaluate(model: DeepMDDQueryDetector, loader: DataLoader[Any], device: torch.device, *,
             frame_steps: tuple[int, ...], precision: str, prefetch: bool,
             augmenter: VideoAugmenter, profile: str, output: Path, preview_clips: int = 3) -> dict[str, Any]:
    model.eval()
    metrics = CoordinateMetrics(frame_steps)
    occluded_metrics = CoordinateMetrics(frame_steps)
    owners: dict[tuple[int, str, int], tuple[tuple[float, int], bool]] = {}
    previews: set[str] = set()
    begin = time.perf_counter()
    with torch.no_grad(), coordinate_compile_scope(model):
        for i, batch in enumerate(coordinate_batches(loader, device, prefetch=prefetch), 1):
            rgb = image_input(batch, device)
            rgb, transformed, audit = augmenter(rgb, batch, epoch=0, profile=profile)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                prediction = model(rgb, batch["timestamps"].to(device, non_blocking=True)).cpu()
            target = {**transformed, "uv": transformed["uv"].cpu(), "position_valid": transformed["position_valid"].cpu()}
            errors = ((prediction - target["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            metrics.add(target, errors)
            occluded = audit.get("artificially_occluded")
            if occluded is not None:
                occluded = occluded.cpu()
                for row, clip_id in enumerate(batch["clip_id"]):
                    for position, frame_index in enumerate(batch["frame_indices"][row].tolist()):
                        if not bool(target["position_valid"][row, position]):
                            continue
                        key = int(batch["frame_step"][row]), clip_id, int(frame_index)
                        owner = abs(position - 15.5), int(batch["start"][row])
                        if key not in owners or owner < owners[key][0]:
                            owners[key] = owner, bool(occluded[row, position])
            clip = batch["clip_id"][0]
            if len(previews) < preview_clips and clip not in previews and batch["frame_step"][0] == 1:
                save_query_preview(output / f"clip-{len(previews):02d}.gif", rgb, prediction, target, occluded)
                previews.add(clip)
            if i % 500 == 0 or i == len(loader):
                print(json.dumps(dict(phase="post_validation", profile=profile, batches=i,
                                      total_batches=len(loader), seconds=time.perf_counter() - begin)), flush=True)
    occluded_metrics.frames = {key: value for key, value in metrics.frames.items() if key in owners and owners[key][1]}
    result = dict(profile=profile, precision=precision, **metrics.report())
    result["artificially_occluded"] = occluded_metrics.report() if occluded_metrics.frames else None
    return result
