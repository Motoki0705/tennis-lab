"""Unique-frame evaluation per FPS, with full and identity-matched common scopes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.model_io.predict import (
    CoordinateModel,
    predict_coordinates,
)
from src.tasks.ball_detection.training.runtime.compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    prediction_batch,
)


def frame_membership_sha256(keys: Iterable[tuple[int, str, int]]) -> str:
    """Identity of the exact FPS/clip/frame set, independent of predictions/order."""
    payload = json.dumps(sorted(set(keys)), separators=(",", ":")).encode()
    return hashlib.sha256(b"mdd-evaluation-frame-membership.v1\n" + payload).hexdigest()


def evaluation_cohorts(data: Dataset[Any]) -> dict[str, str] | None:
    if not isinstance(data, FrozenBallWindowDataset):
        raise ValueError("Coordinate evaluation requires immutable window membership")
    declared = data.manifest.get("evaluation_cohorts")
    if declared is None:
        return None
    if set(declared) != {"existing", "new"}:
        raise ValueError("Evaluation requires explicit existing and new cohorts")
    mapping = {clip: name for name, clips in declared.items() for clip in clips}
    if len(mapping) != sum(len(clips) for clips in declared.values()):
        raise ValueError("Evaluation cohort membership overlaps")
    records = {r["clip"]["clip_id"] for r in data.records}
    if not records <= mapping.keys() or {mapping[name] for name in records} != {"existing", "new"}:
        raise ValueError("Evaluation split must contain both cohorts and assign every eligible clip")
    if any(r["clip"]["split"] == "train" and r["clip"]["clip_id"] in mapping for r in data.manifest["clips"]):
        raise ValueError("Evaluation cohorts cannot contain training clips")
    return {name: mapping[name] for name in records}


class CoordinateMetrics:
    def __init__(self, frame_steps: tuple[int, ...], cohorts: dict[str, str] | None = None) -> None:
        self.frame_steps = frame_steps
        self.cohorts = cohorts
        self.frames: dict[tuple[int, str, int], tuple[tuple[float, int], float, str, bool]] = {}

    def add(self, batch: dict[str, Any], errors: Tensor) -> None:
        for row, clip_id in enumerate(batch["clip_id"]):
            if self.cohorts is not None and clip_id not in self.cohorts:
                raise ValueError("Evaluated clip is outside the declared cohorts")
            step = int(batch["frame_step"][row])
            if step not in self.frame_steps:
                raise ValueError("Evaluation encountered an undeclared frame step")
            common, source = bool(batch["common_evaluation"][row]), str(batch["source"][row])
            start = int(batch["start"][row])
            for position, frame in enumerate(batch["frame_indices"][row].tolist()):
                if not bool(batch["position_valid"][row, position]):
                    continue
                key = step, clip_id, int(frame)
                owner = abs(position - (batch["frame_indices"].shape[1] - 1) / 2), start
                value = float(errors[row, position])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Coordinate error must be finite and nonnegative")
                previous = self.frames.get(key)
                if previous is not None and previous[2:] != (source, common):
                    raise ValueError("Frame has inconsistent source/common evaluation membership")
                if previous is None or owner < previous[0]:
                    self.frames[key] = owner, value, source, common

    def _summary(self, common: bool, source: str | None = None, cohort: str | None = None) -> dict[str, Any]:
        rows = [(key, row) for key, row in self.frames.items()
                if (not common or row[3]) and (source is None or row[2] == source)
                and (cohort is None or (self.cohorts is not None and self.cohorts[key[1]] == cohort))]
        groups = {}
        means = []
        for step in self.frame_steps:
            values = np.asarray([row[1] for key, row in rows if key[0] == step], np.float64)
            group: dict[str, Any] = dict(observed_frames=len(values), available=bool(len(values)))
            for name, value in (("mean_error_px", np.mean(values) if len(values) else None),
                                ("median_error_px", np.median(values) if len(values) else None),
                                ("p95_error_px", np.quantile(values, .95) if len(values) else None)):
                group[name] = float(value) if value is not None else None
            groups[str(step)] = group
            if len(values):
                means.append(group["mean_error_px"])
        return dict(by_frame_step=groups, frame_fps_pairs=len(rows),
                    membership_sha256=frame_membership_sha256(key for key, _ in rows),
                    unique_observed_frames=len({key[1:] for key, _ in rows}),
                    macro_mean_error_px=float(np.mean(means)) if len(means) == len(self.frame_steps) else None)

    def report(self, *, allow_empty: bool = False, sources: list[str] | None = None) -> dict[str, Any]:
        if not self.frames and not allow_empty:
            raise ValueError("Evaluation has no observed coordinates")
        sources = sorted({row[2] for row in self.frames.values()} | set(sources or []))
        result = {}
        for scope, common in (("full", False), ("common", True)):
            result[scope] = self._summary(common)
            result[scope]["by_source"] = {source: self._summary(common, source) for source in sources}
        if self.cohorts is not None:
            for cohort in ("existing", "new"):
                result[cohort] = self._summary(False, cohort=cohort)
                result[cohort]["by_source"] = {source: self._summary(False, source, cohort) for source in sources}
        return dict(schema="mdd_coordinate_evaluation.v1", scopes=result,
                    ownership="closest window centre, then earliest start; counted separately for each FPS",
                    aggregate="equal mean of per-FPS mean source-pixel errors; not a pooled duplicate-frame mean")


def selection_score(report: dict[str, Any], scope: str) -> float:
    if scope == "balanced_existing_new":
        return .5 * selection_score(report, "existing") + .5 * selection_score(report, "new")
    if scope not in {"full", "common", "existing", "new"}:
        raise ValueError("Unknown validation selection scope")
    value = report["scopes"][scope]["macro_mean_error_px"]
    if value is None or not np.isfinite(value):
        raise ValueError(f"Validation scope {scope} has an empty configured FPS; no fallback to another scope")
    return float(value)


def evaluate_coordinates(model: CoordinateModel, loader: DataLoader[Any], device: torch.device,
                         frame_steps: tuple[int, ...], *, precision: str = "fp32", image_prefetch: bool = False) -> dict[str, Any]:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Evaluation precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 evaluation requires a supported CUDA device; no fallback")
    model.eval()
    metrics = CoordinateMetrics(frame_steps, evaluation_cohorts(loader.dataset))
    with torch.no_grad(), coordinate_compile_scope(model):
        for batch in coordinate_batches(loader, device, prefetch=image_prefetch):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                uv = predict_coordinates(model, prediction_batch(batch, device), device).cpu()
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            metrics.add(batch, errors)
    return dict(precision=precision, **metrics.report())
