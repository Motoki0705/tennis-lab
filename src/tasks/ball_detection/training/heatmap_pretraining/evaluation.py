from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.ball_detection.models.mdd_pretrain import MDDDPTDetector
from src.tasks.ball_detection.training.coordinate_compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import CoordinateMetrics
from src.tasks.ball_detection.training.coordinate_images import coordinate_batches
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic

from .objective import heatmap_objective, image_input


class HeatmapMetrics:
    def __init__(self, frame_steps: tuple[int, ...]) -> None:
        self.coordinates = CoordinateMetrics(frame_steps)
        self.frame_steps = frame_steps
        self.frames: dict[tuple[int, str, int], dict[str, Any]] = {}

    def add(self, batch: dict[str, Any], errors: Tensor, confidence: Tensor, losses: Tensor) -> None:
        self.coordinates.add(batch, errors)
        for row, clip in enumerate(batch["clip_id"]):
            for t, frame in enumerate(batch["frame_indices"][row].tolist()):
                if not bool(batch["heatmap_valid"][row, t]):
                    continue
                key = int(batch["frame_step"][row]), clip, int(frame)
                owner = abs(t - 15.5), int(batch["start"][row])
                if key in self.frames and self.frames[key]["owner"] <= owner:
                    continue
                values = [float(errors[row, t]), float(confidence[row, t]), float(losses[row, t])]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite heatmap evaluation")
                self.frames[key] = dict(owner=owner, source=batch["source"][row], common=bool(batch["common_evaluation"][row]),
                                        positive=bool(batch["position_valid"][row, t]), error=values[0], score=values[1], loss=values[2])

    def augment(self, report: dict[str, Any], common: bool, source: str | None = None) -> None:
        losses, f1s = [], []
        for step in self.frame_steps:
            rows = [r for k, r in self.frames.items() if k[0] == step and (not common or r["common"])
                    and (source is None or r["source"] == source)]
            if not rows:
                continue
            positive = sum(r["positive"] for r in rows)
            detected = sum(r["score"] >= .5 for r in rows)
            tp = sum(r["positive"] and r["score"] >= .5 and r["error"] <= 8 for r in rows)
            fp, fn = detected - tp, positive - tp
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            loss = float(np.mean([r["loss"] for r in rows]))
            report["by_frame_step"][str(step)].update(supervised_frames=len(rows), negative_frames=len(rows) - positive,
                heatmap_loss=loss, true_positive=tp, false_positive=fp, false_negative=fn, f1_at_8px=f1,
                precision_at_8px=tp / max(detected, 1), recall_at_8px=tp / max(positive, 1),
                mean_peak_confidence=float(np.mean([r["score"] for r in rows])))
            losses.append(loss)
            f1s.append(f1)
        report["macro_heatmap_loss"] = float(np.mean(losses)) if len(losses) == len(self.frame_steps) else None
        report["macro_f1_at_8px"] = float(np.mean(f1s)) if len(f1s) == len(self.frame_steps) else None

    def report(self) -> dict[str, Any]:
        report: dict[str, Any] = self.coordinates.report()
        report["schema"] = "mdd_heatmap_pretraining_evaluation.v1"
        report["peak_decoding"] = "single global argmax plus log-parabolic subpixel refinement"
        report["detection_threshold"] = .5
        report["matching_radius_source_px"] = 8
        for scope, common in (("full", False), ("common", True)):
            entry = report["scopes"][scope]
            self.augment(entry, common)
            for source, child in entry["by_source"].items():
                self.augment(child, common, source)
        return report


def save_preview(path: Path, rgb: Tensor, probability: Tensor, uv: Tensor, batch: dict[str, Any]) -> None:
    """Fixed validation clip, with GT green / prediction orange and a heatmap panel."""
    import cv2
    from PIL import Image, ImageDraw

    rgb = rgb[0].detach().cpu()
    probability = probability[0].detach().cpu()
    frames = []
    for t in range(32):
        left = Image.fromarray(rgb[t].permute(1, 2, 0).numpy()).resize((480, 270))
        draw = ImageDraw.Draw(left)
        for point, valid, color in [(batch["uv"][0, t], bool(batch["position_valid"][0, t]), "lime"),
                                     (uv[0, t], True, "orange")]:
            if valid:
                x, y = float(point[0]) * 479, float(point[1]) * 269
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        heat = cv2.applyColorMap((probability[t].numpy() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        right = Image.fromarray(heat[..., ::-1]).resize((480, 270))
        canvas = Image.new("RGB", (960, 294), "white")
        canvas.paste(left, (0, 24))
        canvas.paste(right, (480, 24))
        ImageDraw.Draw(canvas).text((5, 5), f"frame {int(batch['frame_indices'][0, t])} | GT green, prediction orange | heatmap 0..1", fill="black")
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def evaluate(model: MDDDPTDetector, loader: DataLoader[Any], device: torch.device, *, frame_steps: tuple[int, ...],
             precision: str, image_prefetch: bool, output: Path, preview_clips: int,
             sigma_ratio: float, gamma: float) -> dict[str, Any]:
    model.eval()
    metrics = HeatmapMetrics(frame_steps)
    previews: set[str] = set()
    started = time.perf_counter()
    with torch.no_grad(), coordinate_compile_scope(model):
        for i, batch in enumerate(coordinate_batches(loader, device, prefetch=image_prefetch), 1):
            rgb = image_input(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(rgb)
            _, per_frame = heatmap_objective(logits, batch, sigma_ratio=sigma_ratio, gamma=gamma)
            probability = logits.float().sigmoid()
            uv, confidence = heatmaps_to_argmax(probability)
            uv = refine_peaks_log_parabolic(probability, uv).cpu()
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            metrics.add(batch, errors, confidence.cpu(), per_frame.cpu())
            clip = batch["clip_id"][0]
            if len(previews) < preview_clips and clip not in previews and batch["frame_step"][0] == 1:
                save_preview(output / f"clip-{len(previews):02d}.gif", rgb, probability, uv, batch)
                (output / f"clip-{len(previews):02d}.json").write_text(json.dumps(dict(clip_id=clip, start=batch["start"][0])))
                previews.add(clip)
            if i % 500 == 0 or i == len(loader):
                print(json.dumps(dict(phase="validation_progress", batches=i, total_batches=len(loader),
                                      seconds=time.perf_counter() - started)), flush=True)
    return dict(precision=precision, **metrics.report())
