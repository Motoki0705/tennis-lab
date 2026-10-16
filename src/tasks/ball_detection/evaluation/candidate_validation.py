"""One candidate-recall path for standard and GAN validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from src.tasks.ball_detection.configuration import validate_candidate_settings
from src.tasks.ball_detection.evaluation.candidate_recall import (
    CandidateRecallCounts,
    candidate_recall_counts,
)
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig

FrameKey = tuple[str, int]


@dataclass(frozen=True)
class FrameRecall:
    """Small CPU record; no heatmaps or GPU tensors survive a batch."""

    priority: tuple[int, int, int]
    source: str
    camera: str
    target: tuple[float, float] | None
    counts: CandidateRecallCounts


def merge_frames(destination: dict[FrameKey, FrameRecall], incoming: Mapping[FrameKey, FrameRecall]) -> None:
    """Select by centre distance, then earlier window/time; never use scores/GT."""
    for key, value in incoming.items():
        previous = destination.get(key)
        if previous is not None:
            if (previous.source, previous.camera, previous.target) != (value.source, value.camera, value.target):
                raise ValueError(f"Validation frame identity changed: {key}")
            if previous.priority == value.priority and previous.counts != value.counts:
                raise ValueError(f"Duplicate validation window has inconsistent candidate outcomes: {key}")
        if previous is None or value.priority < previous.priority:
            destination[key] = value


def _tensor(values: Mapping[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype | None = None) -> Tensor:
    value = values[name]
    if not isinstance(value, Tensor) or value.shape != shape or (dtype is not None and value.dtype != dtype):
        raise ValueError(f"candidate_reference.{name} must be a tensor of shape {shape}, dtype {dtype}")
    return value.detach().cpu()


def _strings(values: Mapping[str, Any], name: str, size: int, *, allow_empty: bool = False) -> list[str]:
    value = values[name]
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(not isinstance(item, str) or (not item and not allow_empty) for item in value)):
        raise ValueError(f"Validation {name} must contain {size} string identities")
    return list(value)


class ValidationCandidateRecall:
    """Micro-count unique validation frames, including distributed sampler repeats.

    Each update uses the native sigmoid grid and the dataset's raw references.
    compute gathers small CPU records across ranks before deduplication; ratios
    are calculated from global counts, never averaged across batches or ranks.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        settings = validate_candidate_settings(settings)
        self.config = BallCandidateConfig(
            max_candidates=settings["max_candidates"],
            nms_kernel=settings["nms_kernel"],
            patch_size=settings["patch_size"],
        )
        self.subpixel_refine: bool = settings["subpixel_refine"]
        self.radius: float = settings["radius_source_px"]
        self.frames: dict[FrameKey, FrameRecall] = {}

    def reset(self) -> None:
        self.frames.clear()

    def update(self, native_heatmaps: Tensor, batch: Mapping[str, Any]) -> None:
        candidates = decode_candidates(native_heatmaps, config=self.config, subpixel_refine=self.subpixel_refine)
        b, t = native_heatmaps.shape[:2]
        reference = batch["candidate_reference"]
        if not isinstance(reference, Mapping):
            raise ValueError("candidate_reference must be an explicit mapping")
        xy = _tensor(reference, "xy", (b, t, 2), torch.float32)
        observed = _tensor(reference, "observed", (b, t), torch.bool)
        ids = _tensor(reference, "frame_id", (b, t), torch.int64)
        starts = _tensor(reference, "window_start", (b,), torch.int64)
        scale = _tensor(reference, "source_scale", (b,), torch.float32)
        sizes = _tensor(batch, "original_size", (b, 2)).float()
        sources = _strings(batch, "source", b)
        namespaces = _strings(reference, "namespace", b)
        cameras = _strings(reference, "camera", b, allow_empty=True)
        if (not torch.isfinite(scale).all() or (scale <= 0).any()
                or not torch.isfinite(sizes).all() or (sizes <= 1).any()
                or (ids < 0).any() or (starts < 0).any()):
            raise ValueError("Validation source scale, image sizes and frame identities must be valid")
        # Store resize uses one width-ratio scale for both axes, not endpoint
        # ratios. This is the same source-pixel mapping as evidence inference.
        candidate_xy = (candidates.coords * ((sizes - 1) / scale[:, None])[:, None, None]).numpy()
        target_xy = (xy / scale[:, None, None]).numpy()
        scores, valid = candidates.scores.numpy(), candidates.valid.numpy()
        for i in range(b):
            for j in range(t):
                counts = candidate_recall_counts(
                    candidate_xy[i, j:j + 1], scores[i, j:j + 1], valid[i, j:j + 1],
                    target_xy[i, j:j + 1], observed[i, j:j + 1].numpy(), radius_px=self.radius,
                )
                target = (float(target_xy[i, j, 0]), float(target_xy[i, j, 1])) if observed[i, j] else None
                record = FrameRecall(
                    priority=(abs(2 * j - (t - 1)), int(starts[i]), j),
                    source=sources[i], camera=cameras[i], target=target, counts=counts,
                )
                merge_frames(self.frames, {(namespaces[i], int(ids[i, j])): record})

    def compute(self, *, require_observed: bool = True) -> dict[str, dict[str, int | float | None]]:
        parts: list[dict[FrameKey, FrameRecall]] = [self.frames]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts = [{} for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(parts, self.frames)
        merged: dict[FrameKey, FrameRecall] = {}
        for part in parts:
            merge_frames(merged, part)
        totals: dict[str, dict[str, int]] = {"": dict.fromkeys(CandidateRecallCounts.__dataclass_fields__, 0)}
        for record in merged.values():
            groups = ["", record.source]
            if record.camera:
                groups.append(f"{record.source}/{record.camera}")
            for group in groups:
                total = totals.setdefault(group, dict.fromkeys(CandidateRecallCounts.__dataclass_fields__, 0))
                for name, value in asdict(record.counts).items():
                    total[name] += value
        if require_observed and totals[""]["observed"] == 0:
            raise ValueError("Validation candidate recall requires at least one single observed ball")
        return {group: CandidateRecallCounts(**counts).report() for group, counts in sorted(totals.items())}


def candidate_log_values(reports: Mapping[str, Mapping[str, int | float | None]]) -> dict[str, float]:
    """Explicit metric names include K/radius; absent denominators emit counts only."""
    aliases = {"recall_at_k": "recall_at_8_20px", "recall_at_1": "recall_at_1_20px"}
    return {
        f"val/{group + '/' if group else ''}candidate_{aliases.get(name, name)}": float(value)
        for group, report in reports.items() for name, value in report.items() if value is not None
    }
