"""Population-based BLCS metrics, independent of loss weights and batch padding."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.blcs.model_io.contracts import (
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
)
from src.tasks.blcs.physics.rollout import rollout_residuals, target_parameters
from src.tasks.blcs.training.losses import projection_errors
from src.utils.physics.ball.surfaces import SURFACE_NAMES
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
)

METRIC_SCHEMA = "blcs.physics_metrics.v1"


class BLCSMetrics:
    """Store per-clip populations; reduce once, including across distributed ranks.

    Scalar errors are Euclidean vector norms. RMSE is sqrt(mean(norm**2)), not
    component RMSE or a mean of batch RMSEs. Undefined populations return NaN
    for Lightning; the saved report converts them to null and records support.
    """

    def __init__(
        self, *, rollout_horizons: Sequence[int] = (1, 3), event_radius_frames: int = 2
    ) -> None:
        self.horizons = tuple(rollout_horizons)
        self.event_radius = event_radius_frames
        self.reset()

    def reset(self) -> None:
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update(
        self,
        prediction: BLCSTrajectoryPrediction,
        target: BLCSTrajectoryTrainingBatch,
        *,
        observed: Tensor,
        scene_ids: Sequence[str] | None = None,
    ) -> dict[str, float]:
        p = denormalize_court_position(prediction.position.float())
        gt = denormalize_court_position(target.position.float())
        velocity = (
            None
            if prediction.velocity is None
            else denormalize_court_velocity(prediction.velocity.float())
        )
        valid, physics = target.loss_mask, target.physics
        error = (p - gt).norm(dim=-1)
        _, pixel, projected, eligible = projection_errors(prediction.position, target)
        residual = (
            None
            if velocity is None
            else rollout_residuals(p, velocity, valid, physics, self.horizons)
        )
        dt = physics.frame_dt().to(p)[:, None, None]
        near = (
            torch.nn.functional.max_pool1d(
                physics.event.ne(0).float().unsqueeze(1),
                self.event_radius * 2 + 1,
                stride=1,
                padding=self.event_radius,
            )[:, 0].bool()
            & valid
        )
        s1, s2 = physics.support(valid, 1), physics.support(valid, 2)
        difference_velocity = ((p[:, 1:] - p[:, :-1]) - (gt[:, 1:] - gt[:, :-1])) / dt
        difference_accel = (
            (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2])
            - (gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2])
        ) / dt.square()
        for b in range(len(p)):
            populations: dict[str, np.ndarray] = {}
            counters: dict[str, int] = {}

            def add(
                key: str,
                values: Tensor,
                destination: dict[str, np.ndarray] = populations,
            ) -> None:
                destination[key] = values.detach().double().cpu().numpy().reshape(-1)

            vb = valid[b]
            add("trajectory", error[b, vb])
            add(
                "velocity_head",
                p.new_empty(0)
                if velocity is None
                else (
                    velocity[b] - denormalize_court_velocity(target.velocity[b].float())
                ).norm(dim=-1)[vb],
            )
            add(
                "position_difference_velocity",
                difference_velocity[b].norm(dim=-1)[s1[b]],
            )
            add(
                "position_difference_acceleration",
                difference_accel[b].norm(dim=-1)[s2[b]],
            )
            add("reprojection", pixel[b].norm(dim=-1)[projected[b]])
            add("reprojection_evaluable", projected[b][eligible[b]].float())
            add("below_ground", (p[b, :, 2] < -0.001)[vb].float())
            add(
                "rollout_position",
                p.new_empty(0)
                if residual is None
                else residual.position[residual.sample == b].norm(dim=-1),
            )
            add(
                "rollout_velocity",
                p.new_empty(0)
                if residual is None
                else residual.velocity[residual.sample == b].norm(dim=-1),
            )
            for h in self.horizons:
                width = max(0, valid.shape[1] - h)
                complete = valid[b, :width].clone()
                for offset in range(1, h + 1):
                    complete &= valid[b, offset : offset + width]
                safe = physics.support(valid, h)[b]
                counters[f"rollout_h{h}_valid_window_count"] = int(complete.sum())
                counters[f"rollout_h{h}_excluded_event_count"] = int(
                    (complete & ~safe).sum()
                )
                counters[f"rollout_h{h}_too_short_clip_count"] = int(
                    valid[b].sum() <= h
                )
                counters[f"rollout_h{h}_no_velocity_clip_count"] = int(velocity is None)
                selected = (
                    None
                    if residual is None
                    else (residual.sample == b) & (residual.horizon == h)
                )
                add(
                    f"rollout_position_h{h}",
                    p.new_empty(0)
                    if residual is None
                    else residual.position[selected].norm(dim=-1),
                )
                add(
                    f"rollout_velocity_h{h}",
                    p.new_empty(0)
                    if residual is None
                    else residual.velocity[selected].norm(dim=-1),
                )
                add(
                    f"rollout_duration_h{h}",
                    p.new_empty(0)
                    if residual is None
                    else residual.duration_s[selected],
                )
                add(
                    f"rollout_substeps_h{h}",
                    p.new_empty(0)
                    if residual is None
                    else residual.simulation_steps[selected],
                )
            view_count = observed[b].bool().sum(dim=0)
            strata = {
                "observed": view_count > 0,
                "missing": view_count == 0,
                "event_near": near[b],
                "free_flight": ~near[b],
            }
            for views in range(7):
                strata[f"views_{views}"] = view_count == views
            for index, name in enumerate(SURFACE_NAMES):
                strata[f"surface_{name}"] = torch.full_like(
                    vb, int(physics.surface[b]) == index
                )
            length = int(physics.clip_frames[b])
            for name, match in (
                ("length_lt128", length < 128),
                ("length_128_511", 128 <= length < 512),
                ("length_ge512", length >= 512),
            ):
                strata[name] = torch.full_like(vb, match)
            bounces = int(physics.bounce_count[b])
            for count in range(3):
                strata[f"bounces_{count}{'plus' if count == 2 else ''}"] = (
                    torch.full_like(vb, min(bounces, 2) == count)
                )
            missing = ((view_count == 0) & vb).cpu().numpy()
            gap_length: NDArray[np.int64] = np.zeros(len(missing), dtype=np.int64)
            start = 0
            while start < len(missing):
                if not missing[start]:
                    start += 1
                    continue
                end = start + 1
                while end < len(missing) and missing[end]:
                    end += 1
                gap_length[start:end] = end - start
                start = end
            gaps = torch.as_tensor(gap_length, device=vb.device)
            strata.update(
                gap_1_4=(gaps > 0) & (gaps <= 4),
                gap_5_15=(gaps >= 5) & (gaps <= 15),
                gap_ge16=gaps >= 16,
            )
            for name, mask in strata.items():
                add(f"trajectory__{name}", error[b, vb & mask])
            if prediction.physics_parameters is not None:
                param_error = (
                    prediction.physics_parameters[b].float()
                    - target_parameters(physics)[b]
                )
                add("parameter_drag", param_error[:, 0].abs()[vb])
                add("parameter_wind", param_error[:, 1:4].norm(dim=-1)[vb])
                add("parameter_effective_magnus", param_error[:, 4:].norm(dim=-1)[vb])
            scene = (
                scene_ids[b] if scene_ids is not None else f"local-{len(self.records)}"
            )
            indices = physics.frame_index[b, vb].cpu().tolist()
            key = f"{scene}:{indices}" if scene_ids is not None else None
            self.records.append(
                {
                    "key": key,
                    "scene": scene,
                    "populations": populations,
                    "counters": counters,
                }
            )
        return {}

    def _merged(self) -> list[dict[str, Any]]:
        records = self.records
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[Any] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, records)
            records = [record for rank in gathered for record in rank]
        # DistributedSampler may repeat the last sample to equalize ranks.
        unique, seen = [], set()
        for record in records:
            key = record["key"]
            if key is not None and key in seen:
                continue
            unique.append(record)
            if key is not None:
                seen.add(key)
        return unique

    def compute(self) -> dict[str, float]:
        values: dict[str, list[np.ndarray]] = defaultdict(list)
        clips: dict[str, list[np.ndarray]] = defaultdict(list)
        counters: dict[str, int] = defaultdict(int)
        for record in self._merged():
            for key, count in record["counters"].items():
                counters[key] += count
            for key, array in record["populations"].items():
                values[key].append(array)
            clips[record["scene"]].append(record["populations"]["trajectory"])
        result: dict[str, float] = {}
        units = {
            "trajectory": "m",
            "velocity_head": "mps",
            "position_difference_velocity": "mps",
            "position_difference_acceleration": "mps2",
            "reprojection": "px",
            "rollout_position": "m",
            "rollout_velocity": "mps",
            "parameter_drag": "per_m",
            "parameter_wind": "mps",
            "parameter_effective_magnus": "per_s",
        }
        # Keep core undefined metrics present even for an all-padding batch.
        for key in units:
            values.setdefault(key, [])
        for key, arrays in values.items():
            array = np.concatenate(arrays) if arrays else np.empty(0)
            count = len(array)
            result[f"{key}_count"] = float(count)
            if key in {"below_ground", "reprojection_evaluable"}:
                result[f"{key}_rate"] = float(np.mean(array)) if count else float("nan")
                continue
            if key.startswith(("rollout_duration_", "rollout_substeps_")):
                result[f"{key}_mean"] = float(np.mean(array)) if count else float("nan")
                continue
            base = key.split("__")[0]
            if base.startswith("rollout_position_h"):
                unit = "m"
            elif base.startswith("rollout_velocity_h"):
                unit = "mps"
            else:
                unit = units[base]
            suffix = "__" + key.split("__", 1)[1] if "__" in key else ""
            base = base if not suffix else "trajectory"
            result[f"{base}_rmse_{unit}{suffix}"] = (
                float(np.sqrt(np.mean(array**2))) if count else float("nan")
            )
            result[f"{base}_mean_{unit}{suffix}"] = (
                float(np.mean(array)) if count else float("nan")
            )
            result[f"{base}_p95_{unit}{suffix}"] = (
                float(np.quantile(array, 0.95)) if count else float("nan")
            )
        rmses = [
            float(np.sqrt(np.mean(np.concatenate(arrays) ** 2)))
            for arrays in clips.values()
            if sum(map(len, arrays))
        ]
        result["clip_trajectory_rmse_mean_m"] = (
            float(np.mean(rmses)) if rmses else float("nan")
        )
        result["clip_count"] = float(len(rmses))
        result.update({key: float(count) for key, count in counters.items()})
        return result

    def report(self) -> dict[str, Any]:
        return {
            "schema": METRIC_SCHEMA,
            "metrics": {
                key: value if np.isfinite(value) else None
                for key, value in self.compute().items()
            },
        }
