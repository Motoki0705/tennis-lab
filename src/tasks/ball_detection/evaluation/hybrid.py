"""Ungated source-pixel localization and independently scored presence decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.ball_detection.evaluation.coordinates import CoordinateMetrics
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector
from src.tasks.ball_detection.training.hybrid.configuration import validate_threshold
from src.tasks.ball_detection.training.hybrid.objective import hybrid_objective
from src.tasks.ball_detection.training.runtime.compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
)
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic


class HybridMetrics:
    def __init__(self, frame_steps: tuple[int, ...], *, presence_threshold: float) -> None:
        validate_threshold(presence_threshold)
        self.coordinates = CoordinateMetrics(frame_steps)
        self.frame_steps, self.threshold = frame_steps, presence_threshold
        self.frames: dict[tuple[int, str, int], dict[str, Any]] = {}

    def add(self, batch: dict[str, Any], errors: Tensor, probabilities: Tensor,
            heatmap_losses: Tensor, presence_losses: Tensor) -> None:
        self.coordinates.add(batch, errors)
        for row, clip in enumerate(batch["clip_id"]):
            step = int(batch["frame_step"][row])
            if step not in self.frame_steps:
                raise ValueError("Presence evaluation encountered an undeclared FPS")
            for t, frame in enumerate(batch["frame_indices"][row].tolist()):
                if not bool(batch["heatmap_valid"][row, t]):
                    continue
                key = step, clip, int(frame)
                owner = abs(t - 15.5), int(batch["start"][row])
                positive = bool(batch["position_valid"][row, t])
                source, common = str(batch["source"][row]), bool(batch["common_evaluation"][row])
                previous = self.frames.get(key)
                if previous and (previous["positive"], previous["source"], previous["common"]) != (positive, source, common):
                    raise ValueError("Presence frame has inconsistent teacher/source/common membership")
                values = [float(probabilities[row, t]), float(heatmap_losses[row, t]), float(presence_losses[row, t])]
                if not np.isfinite(values).all() or not 0 <= values[0] <= 1 or min(values[1:]) < 0:
                    raise ValueError("Invalid presence probability or loss")
                if previous is None or owner < previous["owner"]:
                    self.frames[key] = dict(owner=owner, positive=positive, source=source, common=common,
                        predicted=values[0] >= self.threshold, heatmap_loss=values[1], presence_loss=values[2])

    def augment(self, report: dict[str, Any], common: bool, source: str | None = None) -> None:
        rows = [(key, value) for key, value in self.frames.items()
                if (not common or value["common"]) and (source is None or value["source"] == source)]
        rates: dict[str, list[float]] = {name: [] for name in ("false_positive_rate", "false_negative_rate", "heatmap_loss", "presence_loss")}
        for step in self.frame_steps:
            values = [value for key, value in rows if key[0] == step]
            positives = sum(v["positive"] for v in values)
            negatives = len(values) - positives
            fp = sum(v["predicted"] and not v["positive"] for v in values)
            fn = sum(not v["predicted"] and v["positive"] for v in values)
            stats = dict(supervised_frames=len(values), positive_frames=positives, negative_frames=negatives,
                         true_positive=positives - fn, true_negative=negatives - fp,
                         false_positive=fp, false_negative=fn,
                         false_positive_rate=fp / negatives if negatives else None,
                         false_negative_rate=fn / positives if positives else None,
                         heatmap_loss=float(np.mean([v["heatmap_loss"] for v in values])) if values else None,
                         presence_loss=float(np.mean([v["presence_loss"] for v in values])) if values else None)
            report["by_frame_step"][str(step)]["presence"] = stats
            for name in rates:
                if stats[name] is not None:
                    rates[name].append(float(stats[name]))
        report["presence_frame_fps_pairs"] = len(rows)
        report["unique_supervised_frames"] = len({key[1:] for key, _ in rows})
        for name, rate_values in rates.items():
            report[f"macro_{name}"] = float(np.mean(rate_values)) if len(rate_values) == len(self.frame_steps) else None

    def report(self) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("Hybrid evaluation has no supervised frames")
        report: dict[str, Any] = self.coordinates.report(allow_empty=True, sources=sorted({r["source"] for r in self.frames.values()}))
        report.update(schema="mdd_hybrid_evaluation.v1", presence_threshold=self.threshold,
                      position_population="all observed positions; decode heatmap even when predicted absent",
                      presence_population="observed positives and explicitly selected confirmed negatives; uncertain ignored",
                      peak_decoding="global argmax plus log-parabolic subpixel refinement")
        for scope, common in (("full", False), ("common", True)):
            entry = report["scopes"][scope]
            self.augment(entry, common)
            for source, child in entry["by_source"].items():
                self.augment(child, common, source)
        return report


def evaluate_hybrid(model: DeepMDDHybridDetector, loader: DataLoader[Any], device: torch.device, *,
                    frame_steps: tuple[int, ...], presence_threshold: float, loss_config: dict[str, float],
                    precision: str, image_prefetch: bool) -> dict[str, Any]:
    if precision not in {"fp32", "bf16"} or (precision == "bf16" and device.type != "cuda"):
        raise ValueError("Invalid hybrid evaluation precision/device")
    model.eval()
    metrics = HybridMetrics(frame_steps, presence_threshold=presence_threshold)
    with torch.no_grad(), coordinate_compile_scope(model):
        for batch in coordinate_batches(loader, device, prefetch=image_prefetch):
            rgb = image_input(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model(rgb, batch["timestamps"].to(device))
            _, losses = hybrid_objective(output, batch, loss_config=loss_config)
            heatmaps = output.heatmap_logits.float().sigmoid()
            uv, _ = heatmaps_to_argmax(heatmaps)
            uv = refine_peaks_log_parabolic(heatmaps, uv).cpu()
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            metrics.add(batch, errors, output.presence_logits.float().sigmoid().cpu(),
                        losses["heatmap_frames"].cpu(), losses["presence_frames"].cpu())
    return dict(precision=precision, **metrics.report())
